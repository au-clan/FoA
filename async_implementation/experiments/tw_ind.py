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
log_folder = f"logs_recent/textworld{day}/{hour}/treasure/"
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
    "use_azure": False,
}

# available models : gpt-35-turbo-0125, gpt-4-0125-preview, gpt-4-0613, gpt-4-turbo-2024-04-09

models = {
    "step": "gpt-3.5-turbo-0125",
    "eval": "gpt-3.5-turbo-0125",
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
    agents = agents.copy()

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
    
    dead_agents = [0] * num_agents

    # Batcher
    step_batcher = BatchingAPI(api, batch_size=num_agents, timeout=2, model=models["step"], tab="step")
    
    # Log - Setup
    log = {}
    log[puzzle_idx] = {"environment": {"File":puzzle, "Initial observation":agents[0].strip_obs(agents[0].observations[0])}}
    log[puzzle_idx].update({f"Agent {i}": {} for i in range(num_agents)})

    num_steps = foa_options["num_steps"]
    for step in range(num_steps):

        print(f"Step {step}")

        # mutation phase
        step_coroutines = []
        for (agent_id, agent), is_dead in zip(enumerate(agents), dead_agents):
            if is_dead:
                continue
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
            dead_agents = [(agent.terminal and not agent.has_won()) for agent in agents]
        

    
    return agents, log

    

async def run(run_options: dict, foa_options: dict, log_file:str):

    # Value cache : Not needed for TextWorld (no evaluation)
    # value_cache = {}

    # Get the data for each puzzle
    puzzle_idxs, puzzles = dataset.get_data(run_options["set"], challenge=run_options["challenge"], level=run_options["level"])

    ### Debugging
    #n = 3; puzle_idxs, puzzles = puzzle_idxs[:n], puzzles[:n]

    # Run FoA for each puzzle experiment
    puzzle_coroutines = []
    for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles):
        puzzle_coroutines.append(foa_tw(api, puzzle_idx, puzzle, foa_options, seed=run_options["seed"]))
    results = await asyncio.gather(*puzzle_coroutines)
    puzzle_agents, logs = zip(*results)

    # Compute metrics
    n_success = 0
    for puzzle in puzzle_agents:
        verifications = [agent.has_won() for agent in puzzle]
        if any(verifications):
            n_success += 1
    accuracy = n_success * 100 / len(results)
    metrics = {"Accuracy": accuracy}

   # Merge logs of all puzzles
    merged_log = {}
    logs = [log for log in logs]
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
    args.add_argument("--challenge", type=str, choices=["simple", "cooking", "coin", "treasure"], default="simple")
    args.add_argument("--num_agents", type=int, default=5)
    args.add_argument("--level", type=int, default=None)
    args.add_argument("--num_steps", type=int, default=10)
    args.add_argument('--repeats', type=int, default=1)
    args.add_argument('--send_email', action=argparse.BooleanOptionalAction)
    args = args.parse_args()
    return args

async def main():
    args = parse_args()

    # Select parameters
    set = args.set                        # Set of data to be used
    challenge = args.challenge            # Textworld challenge
    level = args.level                    # Level of the challenge
    num_agents = args.num_agents          # Number of agents
    num_steps = args.num_steps            # Max allowed steps
    repeats = args.repeats                # Number of times to repeat the whole experiment
    send_email = args.send_email          # Send email notification
    

    # Initial name of the final log file (just name, no path)
    if level is None:
        log_file_ = f"{set}-set_{num_agents}agents_{num_steps}steps_{challenge}_independent.json"
    else:
        log_file_ = f"{set}-set_{num_agents}agents_{num_steps}steps_{level}level_{challenge}_independent.json"

    # Organizing FoA arguments
    foa_options = {
        "num_agents": num_agents,
        "num_steps": num_steps,
    }

    # Setting a run message
    run_message = f"""Run options :
        task : TextWorld (independent agents)
        set : {set}
        num_agents : {num_agents}
        num_steps : {num_steps}
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
            "level":level,
        }

        # Run the experiment
        metrics = await run(run_options, foa_options, log_file)

        # Compute accuracy TODO
        for metric, value in metrics.items():
            print(f"{metric.replace("_", " ").capitalize()} : {value:.3f}")
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
            message = '\n'.join(f'{key.upper()}: {value}' for key, value in metrics.items()) + f"\nCost : {cost:.2f}"
            try:
                email_notification(subject=subject, message=message)
                print("Email sent successfully.")
            except:
                print("Email failed to send.")
                pass




if __name__ == "__main__":
    asyncio.run(main())
