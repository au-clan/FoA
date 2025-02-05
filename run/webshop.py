import argparse
import json
import asyncio
import os
import random
import numpy as np

from tqdm import tqdm

from datetime import datetime
from diskcache import Cache

import sys
sys.path.append(os.getcwd())
from src.agents.ws import WebShopAgent
from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from data.data import WebShopData
from utils import create_folder, create_box, update_actual_cost

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Folder in which logs will be saved
time = datetime.now()
day = time.strftime("%d-%m")
hour = time.strftime("%H")
log_folder = f"logs_recent/ws/{datetime.now().strftime("%m-%d/%H/%M")}/" # Folder in which logs will be saved 
#log_folder = f"logs_recent/webshop/{day}/{hour}/"
create_folder(log_folder)

# According to ReAct
step_api_config = {
    "max_tokens": 100,
    "temperature": 1,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "request_timeout": 90,
    "stop": ["\n"],
}

eval_api_config = {
    "max_tokens": 100,
    "temperature": 1,
    "request_timeout": 90,
    "stop": None,
}

# Models
## "gpt-3.5-turbo-0125"
## "gpt-4-0613"
## "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
## "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

model = "gpt-3.5-turbo-0125"
provider = "TogetherAI" if "meta" in model else "OpenAI"
models = {
    "step": {"model_name":model, "provider":provider},
    "eval": {"model_name":model, "provider":provider},
}

api = API(step_api_config, models=models.values(), resources=2, verbose=True)

# Setting up the data
dataset = WebShopData()

def resample(agent_records, n, agent_ids, current_step, gamma=0.5, method="linear"):
    """
    Linear resampling of agents based on their rewards
    Input:
        - agents: list of current agents
        - n: number of agents to resample
    Output:
        - resampled_agents: list of resampled agents
    """

    assert n == len(agent_ids), f"Number of agents ({n}) and number of agent ids ({len(agent_ids)}) do not match."
    # Computing discount factors
    gammas = np.array([gamma ** (current_step-step) for _, step in agent_records])

    # Compute probabilities
    eps = 1e-6
    values = np.array([agent.values[-1] for agent, _ in agent_records])
    if method == "linear_filtered":
        values[values < values.max() * 0.5] = 0
    decayed_values = values * gammas + eps
    probs = decayed_values / np.sum(decayed_values)

    if method == "max_unique":
        max_prob = probs.max()
        probs /= max_prob
        first_one_idx = np.argmax(probs)
        probs = np.zeros_like(probs)
        probs[first_one_idx] = 1


    # Resample indices with consistency
    random_seed = agent_records[0][0].random_seed
    np.random.seed(random_seed)
    
    resampled_indices = np.random.choice(range(len(agent_records)), size=n, replace=True, p=probs).tolist()

    # Get resampled agents
    random.seed(random_seed)
    new_random_seeds = [random.randint(1, 1000) for _ in range(len(agent_records))]
    resampled_agents = []
    for indice, id in zip(resampled_indices, agent_ids):
        resampled_agents.append(agent_records[indice][0].clone(random_seed=new_random_seeds[indice], id=id))
    
    for agent in resampled_agents:
        assert agent.env.sessions != {}, "Empty session in resampled agent "

    assert len(resampled_agents) == n, f"Returned {len(resampled_agents)}/{n} requested agents "
    return resampled_agents

async def foa_ws(api, puzzle_idx, puzzle, foa_options, value_cache,seed, barrier):

    env_id = puzzle_idx
    n_evaluations = foa_options["n_evaluations"]
    batching = bool(foa_options["batching"])
    caching = bool(foa_options["caching"])
    
    num_agents = foa_options["num_agents"]
    agents = []
    for i in range(num_agents):
        agents.append(WebShopAgent(env_id, random_seed=i, id=i, prompting=foa_options["prompting"]))
    agents_record = []
    terminal_agents = []

    # Batcher
    step_batcher = BatchingAPI(api, batch_size=num_agents if batching else 1, timeout=120, model=models["step"]["model_name"], tab="step")
    eval_batcher = BatchingAPI(api, batch_size=num_agents*n_evaluations if batching else 1, timeout=120, model=models["eval"]["model_name"], tab="eval")

    # Log - Setup
    log = {}
    log[puzzle_idx] = {"environment": {"File":puzzle, "Initial observation":agents[0].observations[0]}}
    log[puzzle_idx].update({f"Agent {agent.id}": {} for agent in agents})

    num_steps = foa_options["num_steps"]
    k = foa_options["k"]
    solution_found = False
    for step in range(num_steps):

        if solution_found:
            await barrier.wait()
            continue

        print(f"Step {step} (env_id {env_id})")
        
        # mutation phase
        step_coroutines = []
        for agent in agents:
            if not agent.terminal:
                step_coroutines.append(agent.step(step_batcher, namespace=(puzzle_idx, f"Agent: {agent.id}", f"Step: {step}")))
        await asyncio.gather(*step_coroutines)

        # Log - Steps
        for agent in agents:
            log[puzzle_idx][f"Agent {agent.id}"][f"Step {step}"] = {"Latest Observation": agent.observations[-1], "Action history": agent.action_history.copy(), "Latest reward": agent.rewards[-1], "Terminal": agent.terminal}
        
        
        terminal_agents.extend([agent for agent in agents if agent.terminal])
        agents = [agent for agent in agents if not agent.terminal]
        assert len(step_batcher.futures) == 0, f"Step batcher futures not empty: {len(step_batcher.futures)}"
        assert len(eval_batcher.futures) == 0, f"Eval batcher futures not empty: {len(eval_batcher.futures)}"
        step_batcher.batch_size=len(agents)
        eval_batcher.batch_size=len([agent for agent in agents if agent.observations[-1]!="Invalid action!"]*n_evaluations)

        if len(agents) == 0:
            assert len(terminal_agents) == num_agents, f"Irregular break (env_id {env_id}, step {step}): {len(terminal_agents)}/{num_agents} terminal."
            solution_found = True
            await barrier.wait()
            continue
        
        # Synchronize experiments
        await barrier.wait()
        
        # Selection phase
        if 0 < step < num_steps -1 and k>0 and step % k == 0:
            # Evaluation
            value_coroutines = []
            for agent in agents:
                value_coroutines.append(agent.evaluate(eval_batcher, value_cache, n=n_evaluations, namespace=(puzzle_idx, f"Agent: {agent.id}", f"Step: {step}"), caching=caching))
            await asyncio.gather(*value_coroutines)

            agents_record+=[(agent.clone(), step) for agent in agents if agent.values[-1] > 0]
        
            for agent, _ in agents_record:
                assert agent.observations[-1] != "Invalid action!", f"Invalid action in agent {agent.id} at step {step} taken to records in env_id {env_id}\nEvaluate prompt:\n{agent.get_complete_prompt(type='eval')}\nAgent values : {agent.values}"
                assert agent.env.sessions != {}, "Empty session in records agent "

            
            # Resampling
            if len(agents_record) > 0:
                agent_ids = [agent.id for agent in agents]
                agents = resample(agents_record, len(agents), gamma=foa_options["backtrack"], agent_ids=agent_ids, current_step=step, method=foa_options["resampling_method"])

                # Log - Resampling
                for agent in agents:
                    assert agent.observations[-1] != "Invalid action!", f"Invalid action in agent {agent.id} at step {step} resampled in env_id {env_id}"
                    log[puzzle_idx][f"Agent {agent.id}"][f"Step {step}"].update({"Resampling":{"Latest Observation": agent.observations[-1], "Action history": agent.action_history.copy(), "Latest reward": agent.rewards[-1], "Latest value": agent.values[-1],"Terminal": agent.terminal}})
        
    all_agents = agents + terminal_agents
    
    assert len(all_agents) == num_agents, f"{len(all_agents)}/{num_agents} returned"
    return all_agents, log

async def run(run_options: dict, foa_options: dict, log_file:str):

    value_cache = {}


    # Get the data for each puzzle
    puzzle_idxs, puzzles = dataset.get_data(run_options["set"])

    ### Debugging
    # end = 2
    # puzzle_idxs, puzzles = puzzle_idxs[:end], puzzles[:end]

    # Barriers for each puzzle experiment
    barrier = asyncio.Barrier(len(puzzles))

    print(f"Puzzles to solve : {len(puzzle_idxs)}")
    # Run FoA for each puzzle experiment
    puzzle_coroutines = []
    for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles):
        puzzle_coroutines.append(foa_ws(api, puzzle_idx, puzzle, foa_options, value_cache=value_cache, seed=run_options["seed"], barrier=barrier))
    results = await asyncio.gather(*puzzle_coroutines)
    puzzle_agents, logs = zip(*results)

    ### Sync version just for debugging
    # puzzle_agents=[]
    # logs=[]
    # for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles):
    #     pa, l = await foa_ws(api, puzzle_idx, puzzle, foa_options, value_cache=value_cache, seed=run_options["seed"])
    #     puzzle_agents.append(pa)
    #     logs.append(l)


    # Merge logs of all puzzles
    merged_log = {}
    logs = [log for log in logs]
    for log in logs:
        merged_log.update(log)

     # Save merged logs
    with open(log_folder + log_file, 'w+') as f:
        json.dump(merged_log, f, indent=4)

    # Compute metrics
    results = []
    for puzzle in puzzle_agents:
        rewards = [agent.rewards[-1] for agent in puzzle]
        results.append(np.max(rewards))
    mean_reward = np.mean(results)
    percentage_finished = np.mean([1 if reward >0 else 0 for reward in results])
    metrics = {"mean_reward": mean_reward, "percentage_finished": percentage_finished}

    # Update logs with metada
    step_cost = api.cost(tab_name="step")
    evaluation_cost = api.cost(tab_name="eval")
    total_cost = api.cost() 
    merged_log["Info"] = {}
    merged_log["Info"]["Cost"] = {"Step": step_cost, "Evaluation": evaluation_cost, "Total cost": total_cost}
    merged_log["Info"]["Models"] = {"Step": models["step"], "Evaluation": models["eval"]}
    merged_log["Info"]["FoA options"] = foa_options
    merged_log["Info"]["Run options"] = run_options
    merged_log["Info"]["Metrics"] = metrics

    # Save merged logs
    with open(log_folder + log_file, 'w+') as f:
        json.dump(merged_log, f, indent=4)

    # Return puzzle states/agents for each puzzle
    return metrics





#################
### Execution ###
#################

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("--set", type=str, choices=["mini", "train", "validation", "test"], default="mini")
    args.add_argument("--num_agents", type=int, default=10)
    args.add_argument("--backtrack", type=float, default=0.5)
    args.add_argument("--num_steps", type=int, default=5)
    args.add_argument("--resampling", type=str, choices=["linear", "logistic", "max", "percentile", "linear_filtered", "max_unique"], default="linear")
    args.add_argument("--k", type=int, default=1)
    args.add_argument('--debugging', type=int, default=0)
    args.add_argument('--pruning', type=int, default=1)
    args.add_argument('--repeats', type=int, default=1)
    args.add_argument('--caching', type=int, default=1)
    args.add_argument('--batching', type=int, default=1)
    args.add_argument("--n_evaluations", type=int, default=1)
    args.add_argument("--origin_value", type=int, default=0)
    args.add_argument('--prompting', type=str, choices=["act", "react"], default="react")
    args = args.parse_args()
    return args

async def main():
    args = parse_args()

    # Select parameters
    set = args.set                                                  # Set of data to be used
    num_agents = args.num_agents                                    # Number of agents
    k = args.k                                                      # Resampling every <k> steps
    backtrack = args.backtrack                                      # Backtrack decaying coefficient
    origin_value = args.origin_value                                # The evaluation of the origin 
    num_steps = args.num_steps                                      # Max allowed steps
    resampling_method = args.resampling                             # Resampling method
    debugging = args.debugging                                      # Number of puzzles to run
    pruning = args.pruning                                          # Whether to prune or not
    repeats = args.repeats                                          # Number of times to repeat the experiment
    caching = args.caching                                          # Whether to cache the evaluations or not
    batching  = args.batching                                       # Whether to use batching or not
    prompting = args.prompting            # Prompting method
    n_evaluations = args.n_evaluations    # Number of evaluations 

    # Initial name of the final log file (just name, no path)
    log_file_ = f"{set}_{num_agents}a_{num_steps}s_{k}k_{backtrack}b_{n_evaluations}n_{origin_value}or_{prompting}.json"

    # Organizing FoA arguments
    foa_options = {
        "num_agents": num_agents,
        "k": k,
        "origin_value": origin_value,
        "num_steps": num_steps,
        "backtrack": backtrack,
        "resampling_method": resampling_method,
        "prompting": prompting,
        "n_evaluations": n_evaluations,
        "pruning": pruning,
        "caching": caching,
        "batching": batching
    }

    # Setting a run message
    run_message = f"""Run options :
        task : webshop
        set : {set}
        num_agents : {num_agents}
        k : {k}
        num_steps : {num_steps}
        backtrack : {backtrack}
    """
    print("\n"+create_box(run_message)+"\n")

    for seed in range(repeats):
        log_file = log_file_.split(".json")[0] + f"_{seed}.json"

        run_options = {
            "task": "WebShop",
            "set":set,
            "seed":seed,
            "debugging":debugging
        }

        # Run
        metrics = await run(run_options, foa_options, log_file)

        # Compute accuracy
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').upper()} : {value:.3f}")
        print(f"File name : {log_file}\n\n\n\n\n")

        #Update actual cost.
        update_actual_cost(api)

        # Empty api cost so multiple repeats are not cumulative
        api.empty_tabs()




if __name__ == "__main__":
    asyncio.run(main())