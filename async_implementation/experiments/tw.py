import argparse
import json
import asyncio
import os
import random
import numpy as np

from datetime import datetime


import textworld
import textworld.gym
from diskcache import Cache

import sys
sys.path.append(os.getcwd())
from async_implementation.agents.tw import TextWorldAgent
from async_engine.cached_api import CachedOpenAIAPI
from async_engine.batched_api import BatchingAPI
from data.data import TextWorldData
from utils import create_folder, email_notification, create_box, update_actual_cost

# Clear terminal
os.system('cls' if os.name == 'nt' else 'clear')

# Folder in which logs will be saved
time = datetime.now()
day = time.strftime("%d-%m")
hour = time.strftime("%H")
log_folder = f"logs_recent/{day}/{hour}/"
create_folder(log_folder)

assert os.path.exists("./caches/"), "Please run the script from the root directory of the project."
cache = Cache("./caches/tw", size_limit=int(2e10))

step_api_config = eval_api_config = {
    "max_tokens": 100,
    "temperature": 1,
    "top_p": 1,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "request_timeout": 60,
    "use_azure": True,
}

# available models : gpt-35-turbo-0125, gpt-4-0125-preview, gpt-4-0613

models = {
    "step": "gpt-35-turbo-0125",
    "eval": "gpt-35-turbo-0125",
}

api = CachedOpenAIAPI(cache, step_api_config, models=models.values(), resources=2, verbose=True)


# Setting up the data
dataset = TextWorldData()


async def resample(agents, n, gamma=0.5):
    """
    Linear resampling of agents based on their rewards
    Input:
        - agents: list of current agents
        - n: number of agents to resample
    Output:
        - resampled_agents: list of resampled agents
    """
    # Computing discount factors
    max_steps = max([len(agent.rewards) for agent in agents])
    gammas = np.array([gamma ** (max_steps - len(agent.rewards)) for agent in agents])

    # Compute probabilities
    values = (np.array([agent.rewards[-1] for agent in agents]) + 1) * gammas
    probs = values / np.sum(values)
    

    # Resanoke indices with consistency
    random_seed = agents[0].random_seed
    np.random.seed(random_seed)
    resampled_indices = np.random.choice(range(len(agents)), size=n, replace=True, p=probs).tolist()

    # Get resampled agents
    random.seed(random_seed)
    new_random_seeds = [random.randint(1, 1000) for _ in range(len(agents))]
    cloning_coroutines = [agents[i].clone(new_random_seeds[i]) for i in resampled_indices]
    resampled_agents = await asyncio.gather(*cloning_coroutines)
    return resampled_agents

async def foa_tw(api, puzzle_idx, puzzle, foa_options, seed):
    
    # Register the games and request the admissible commands
    request_infos = textworld.EnvInfos(admissible_commands=True, verbs=True, command_templates=True, won=True, max_score=True)

    # Register the game
    game_file = f"data/datasets/tw_games/{puzzle}"
    env_id = textworld.gym.register_game(game_file, request_infos)


    num_agents = foa_options["num_agents"]
    agents = []
    for i in range(num_agents):
        agents.append(TextWorldAgent(env_id, random_seed=i))
    agents_record = agents[:1].copy()

    # Batcher
    step_batcher = BatchingAPI(api, batch_size=num_agents, timeout=2, model=models["step"], tab="step")
    
    # Log - Setup
    log = {}
    log[puzzle_idx] = {"environment": {"File":puzzle, "Initial observation":agents[0].strip_obs(agents[0].observations[0])}}
    log[puzzle_idx].update({f"Agent {i}": {} for i in range(num_agents)})

    num_steps = foa_options["num_steps"]
    k = foa_options["k"]
    for step in range(num_steps):

        # mutation phase
        step_coroutines = []
        for agent_id, agent in enumerate(agents):
            step_coroutines.append(agent.step(step_batcher, namespace=(puzzle_idx, f"Agent: {agent_id}", f"Step: {step}")))
        await asyncio.gather(*step_coroutines)

        # Log - Steps
        for agent_id, agent in enumerate(agents):
            log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"] = {"Latest Observation": agent.observations[-1], "Action history": agent.action_history.copy(), "Rewards": agent.rewards.copy(), "Terminal": agent.terminal, "Won": agent.has_won()}

        # check for terminal agents
        lost_agents = [agent for agent in agents if (agent.terminal and not agent.has_won())]
        won_agents = [agent for agent in agents if (agent.terminal and agent.has_won())]
        if len(won_agents) > 0:
            # At least an agent, has finished successfully
            break
        elif len(lost_agents) > 0:
            # At least an agent, has finished unsuccessfully
            # TODO: Resample from current agents only or record too ? -> So far, only current.
            agents = [agent for agent in agents if agent not in lost_agents]
            resampled_agents = await resample(agents, len(lost_agents))
            agents.extend(resampled_agents)
        
        agents_record.extend(agents.copy())
        
        #selection phase
        if step < num_steps - 1 and step % k == 0:
            agents = await resample(agents_record, num_agents)

            # Log - Resampling
            for agent_id, agent in enumerate(agents):
                log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Resampling":{"Latest Observation": agent.observations[-1], "Action history": agent.action_history.copy(), "Rewards": agent.rewards.copy(), "Terminal": agent.terminal, "Won": agent.has_won()}})
    
    return agents, log

    

async def run(run_options: dict, foa_options: dict, log_file:str):

    # Value cache : Not needed for TextWorld (no evaluation)
    # value_cache = {}

    # Get the data for each puzzle
    puzzle_idxs, puzzles = dataset.get_data(run_options["set"], run_options["challenge"])

    ### Debugging
    # puzle_idxs, puzzles = puzzle_idxs[:1], puzzles[:1]

    # Run FoA for each puzzle experiment
    puzzle_coroutines = []
    for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles):
        puzzle_coroutines.append(foa_tw(api, puzzle_idx, puzzle, foa_options, seed=run_options["seed"]))
    results = await asyncio.gather(*puzzle_coroutines)

    # Merge logs of all puzzles
    merged_log = {}
    logs = [log for (puzzle, log) in results]
    for log in logs:
        merged_log.update(log)
    
    # Update logs with metada
    step_cost = api.cost(tab_name="step")
    evaluation_cost = api.cost(tab_name="eval")
    total_cost = api.cost() 
    merged_log["Info"] = {}
    merged_log["Info"]["Cost"] = {"Step": step_cost, "Evaluation": evaluation_cost, "Total cost": total_cost}
    merged_log["Info"]["Models"] = {"Step": models["step"], "Evaluation": models["eval"]}
    merged_log["Info"]["FoA options"] = foa_options
    merged_log["Info"]["Run options"] = run_options
    
    # Save merged logs
    with open(log_folder + log_file, 'w+') as f:
        json.dump(merged_log, f, indent=4)

    # Return puzzle states/agents for each puzzle
    puzzle_states = [puzzle for (puzzle, log) in results]
    return puzzle_states





#################
### Execution ###
#################

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--set", type=str, choices=["mini", "train", "validation", "test"], default="mini")
    args.add_argument("--challenge", type=str, choices=["simple", "cooking", "coin", "treasure"], default="cooking")
    args.add_argument("--num_agents", type=int, default=1)
    args.add_argument("--backtrack", type=float, default=0.6)
    args.add_argument("--num_steps", type=int, default=10)
    args.add_argument("--resampling", type=str, choices=["linear", "logistic", "max", "percentile", "linear_filtered"], default="linear")
    args.add_argument("--k", type=int, default=1)
    args.add_argument('--repeats', type=int, default=1)
    args.add_argument('--send_email', action=argparse.BooleanOptionalAction)
    args = args.parse_args()
    return args

async def main():
    args = parse_args()

    # Select parameters
    set = args.set                        # Set of data to be used
    challenge = args.challenge            # Textworld challenge
    num_agents = args.num_agents          # Number of agents
    k = args.k                            # Resampling every <k> steps
    backtrack = args.backtrack            # Backtrack decaying coefficient
    origin_value = 0                      # The evaluation of the origin #TODO: Origin value?
    num_steps = args.num_steps            # Max allowed steps
    resampling_method = args.resampling   # Resampling method
    repeats = args.repeats                # Number of times to repeat the whole experiment
    send_email = args.send_email          # Send email notification
    

    # Initial name of the final log file (just name, no path)
    log_file_ = f"{set}-set_{num_agents}agents_{num_steps}steps_{k}k_{origin_value}origin_{backtrack}backtrack_{resampling_method}-resampling.json"

    # Organizing FoA arguments
    foa_options = {
        "num_agents": num_agents,
        "k": k,
        "origin_value": origin_value,
        "num_steps": num_steps,
        "backtrack": backtrack,
        "resampling_method": resampling_method
    }

    # Setting a run message
    run_message = f"""Run options :
        task : gameof24
        set : {set}
        num_agents : {num_agents}
        k : {k}
        num_steps : {num_steps}
        backtrack : {backtrack}
    """
    print("\n"+create_box(run_message)+"\n")

    for seed in range(repeats):
        log_file = log_file_.split(".json")[0] + f"_{seed}.json"
        
        # Organizing Run arguments
        run_options = {
            "task": "TextWorld",
            "set":set,
            "challenge":challenge,
            "seed":seed,
        }

        # Run the experiment
        results = await run(run_options, foa_options, log_file)


        # Compute accuracy
        n_success = 0
        for puzzle in results:
            verifications = [agent.has_won() for agent in puzzle]
            if any(verifications):
                n_success += 1
        accuracy = n_success * 100 / len(results)
        print(f"Accuracy : {accuracy}\n")
        print(f"File name : {log_file}\n\n\n\n\n")

        # Get current cost for email and update actual cost
        cost = api.cost(verbose=True)
        cost = cost["total_cost"]
        update_actual_cost(api)

        # Empty api cost so multiple repeats are not cumulative
        api.empty_tabs()

        # Send email notification
        if send_email:
            subject = f"{seed+1}/{repeats} :" + log_file
            message = f"Accuracy : {accuracy}\nCost : {cost:.2f}"
            try:
                email_notification(subject=subject, message=message)
                print("Email sent successfully.")
            except:
                print("Email failed to send.")
                pass




if __name__ == "__main__":
    asyncio.run(main())
