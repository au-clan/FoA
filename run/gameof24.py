import asyncio
import os
import json
import random
import argparse

from datetime import datetime

# TODO: Not sure if this is optimal, I didn't know how else to handle the package paths
import sys

sys.path.append(os.getcwd()) # Project root!!

from async_engine.api import API
from async_engine.batched_api import BatchingAPI
from src.agents.gameof24 import GameOf24Agent
from src.states.gameof24 import GameOf24State
from src.resampling.resampler import Resampler
from data.data import GameOf24Data
from utils import create_folder, create_box, update_actual_cost

log_folder = f"logs_recent/gameof24/{datetime.now().strftime("%m-%d/%H/%M")}/" # Folder in which logs will be saved 
create_folder(log_folder)

step_api_config = eval_api_config = {
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 1,
    "request_timeout": 120,
    "top_k": 50
}

model = "gpt-4-0613"
provider = "TogetherAI" if "meta" in model else "OpenAI"

models = {
    "step": {"model_name":model, "provider":provider},
    "eval": {"model_name":model, "provider":provider},
}

api = API(eval_api_config, models=models.values(), resources=2, verbose=False)


# Setting up the data
dataset = GameOf24Data()



# ToDo: this should probably be moved to its own file
# for now I'm keeping it here, for easier debugging
async def foa_gameof24(api, puzzle_idx, puzzle, foa_options, barrier, seed, value_cache):
    num_agents = foa_options["num_agents"]
    num_steps = foa_options["num_steps"]
    caching = bool(foa_options["caching"])
    batching = bool(foa_options["batching"])
    pruning = bool(foa_options["pruning"])

    # Use batching API
    step_batcher = BatchingAPI(api, batch_size=num_agents if batching else 1, timeout=2, model=models["step"]["model_name"], tab="step")
    eval_batcher = BatchingAPI(api, batch_size=num_agents*3 if batching else 1, timeout=2, model=models["eval"]["model_name"], tab="eval")

    # Set randomness
    randomness = puzzle_idx + seed
    random.seed(randomness)

    resampler = Resampler(randomness)

    # Set up log
    log = {}
    log[puzzle_idx] = {"puzzle": puzzle}
    log[puzzle_idx].update({f"Agent {i}": {} for i in range(num_agents)})


    # State identifier shows the step and the agent where the state was visited eg. 0.1 means step 0, agent 1
    state_records = [] # List of states [(state_identifier, state_value, state)]
    state_records.append(("INIT", foa_options["origin_value"], GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000))))

    # Set up states
    states = []

    for _ in range(num_agents):
        states.append(GameOf24State(puzzle=puzzle, current_state=puzzle, steps=[], randomness=random.randint(0, 1000)))

    solution_found = False ### DEBUG: Just for now until I figure out something better
    for step in range(num_steps):

        print(f"Step {step} : Stepping")

        ### DEBUG: Just for now until I figure out something better
        if solution_found:
            await barrier.wait()
            continue

        # Log - Set up log of each agent for current step
        for agent_id in range(num_agents):
            log[puzzle_idx][f"Agent {agent_id}"].update({f"Step {step}": {}})

        # Step : make one step for each state
        agent_coroutines = []
        for agent_id, state in enumerate(states):
            agent_coroutines.append(GameOf24Agent.step(state, step_batcher, namespace=(puzzle_idx, f"Agent: {agent_id}", f"Step : {step}")))
        states = await asyncio.gather(*agent_coroutines)

        # Log - Steps
        for agent_id, state in enumerate(states):
            log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Step": f"{' -> '.join(state.steps)}"})

        # Depreciation : old state values are decayed by the backtrack coefficient
        state_records = [(idx, value*foa_options["backtrack"], state) for idx, value, state in state_records]

        # After each step, the api should be empty
        assert len(step_batcher.futures) == 0, f"API futures should be empty, but are {len(step_batcher.futures)}"

        # Update state records
        foa_remaining_steps = num_steps - (step + 1)
        task_required_steps = 0
        state_records = [(idx, value, state) for idx, value, state in state_records if  foa_remaining_steps >= task_required_steps - len(state.steps)] # Remove states that cannot finish in time
        state_records = [(idx, value, state) for idx, value, state in state_records if value>0] # Remove states with no value


        # Verification : After each step we verify if the answer is found and if so we break    
        verifications = [GameOf24Agent.verify(state) for state in states]   # {"r":1} Finished correctly
        if {"r":1} in verifications:                                        # {"r":-1} Finished incorrectly / "Ivalid state"
            ### DEBUG: Just for now until I figure out something better - Normally you want to break here
            solution_found = True
            await barrier.wait()
            continue                                                        # {"r":0} Not finished

        # Pruning
        temp_state_records = [(idx, value, state) for idx, value, state in state_records if state.steps!=[]]
        invalid_state_indices = [i for i, verification in enumerate(verifications) if verification["r"] == -1]
        invalids_resolved = False
        
        if pruning:
            if len(temp_state_records) > 0:
                # If there are eligible + evaluated states.
                new_states, pruned_indices = resampler.resample(temp_state_records, len(invalid_state_indices), foa_options["resampling_method"])
                states = [new_states.pop(0) if i in invalid_state_indices else state for i, state in enumerate(states)]
                invalids_resolved = True
            else:
                # If there are no eligible + evaluated states.
                pruned_indices = [None] * len(invalid_state_indices)

            # Log - Pruning
            for agent_id in range(num_agents):
                if agent_id in invalid_state_indices:
                    pruned_indice = pruned_indices.pop(0)
                    if pruned_indice is None:
                        log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Pruning": "NA"})
                    else:
                        log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Pruning" : {"Idx":temp_state_records[pruned_indice][0], "Pruned state": temp_state_records[pruned_indice][2].current_state, "Value": temp_state_records[pruned_indice][1],}})
                else:
                    log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Pruning": None})

        # Synchronize experiments
        await barrier.wait()

        # Resampling : every k steps, evaluate and resample
        if step < num_steps - 1 and foa_options["k"]>0 and step % foa_options["k"] == 0:

            # Evaluation : each of the current states is given a value
            value_coroutines = []
            for agent_id, state in enumerate(states):
                value_coroutines.append(GameOf24Agent.evaluate(state=state, api=eval_batcher, namespace=(puzzle_idx, f"Agent: {agent_id}", f"Step : {step}"), value_cache=value_cache, caching=caching))
            values = await asyncio.gather(*value_coroutines)

            assert len(eval_batcher.futures) == 0, f"API futures should be empty, but are {len(eval_batcher.futures)}"

            # Update records
            for i, (state, value) in enumerate(zip(states, values)):
                if i not in invalid_state_indices or invalids_resolved:
                    state_records.append((f"{i}.{step}", value, state))
            
            # Log - Evaluation
            for agent_id, value in enumerate(values):
                log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Evaluation": value})

            # Resampling
            states, resampled_indices = resampler.resample(state_records, num_agents, foa_options["resampling_method"])
            
            # Log - Resampling
            for agent_id, resampled_idx in enumerate(resampled_indices):
                log[puzzle_idx][f"Agent {agent_id}"][f"Step {step}"].update({"Resampling": {"Idx":state_records[resampled_idx][0], "Resampled state": state_records[resampled_idx][2].current_state, "Value": state_records[resampled_idx][1],}})

    verifications = [GameOf24Agent.verify(result) for result in states]
    log[puzzle_idx]["Input"] = puzzle
    log[puzzle_idx]["Verifications"] = verifications

    return states, log


async def run(run_options: dict, foa_options: dict, log_file: str):
    """
    Inputs
        difficulty: Selects the starting index
        sample_size: Selects the number of experiments to run
    """

    game_coroutines = []
    log = {}
    seed = run_options["seed"]

    # Value cache
    value_cache = {}

    # Get the data for each puzzle
    puzzle_idxs, puzzles = dataset.get_data(run_options["set"])

    ### Debugging
    if run_options["debugging"] > 0:
        n_puzzles = run_options["debugging"]
        puzzle_idxs, puzzles = puzzle_idxs[:n_puzzles], puzzles[:n_puzzles]

    # Barriers for each puzzle experiment
    barrier = asyncio.Barrier(len(puzzles))

    # Run FoA for each puzzle
    for puzzle_idx, puzzle in zip(puzzle_idxs, puzzles):
        game_coroutines.append(foa_gameof24(api, puzzle_idx, puzzle, foa_options, barrier, seed, value_cache))
    results = await asyncio.gather(*game_coroutines)

    # Merge logs for each run
    logs = [log for (game, log) in results]
    for l in logs:
        log.update(l)

    log["Info"] = {}
    log["Info"]["Cost"] = {
        "Step": api.cost(tab_name="step"),
        "Evaluation": api.cost(tab_name="eval"),
        "Total cost": api.cost()
    }
    log["Info"]["Tokens"] = {
        "Step": api.cost(tab_name="step", report_tokens=True),
        "Evaluation": api.cost(tab_name="eval", report_tokens=True),
        "Total cost": api.cost(report_tokens=True)
    }
    log["Info"]["Models"] = {"Step": models["step"], "Evaluation": models["eval"]}
    log["Info"]["FoA options"] = foa_options
    log["Info"]["Run options"] = run_options

    # Save merged logs
    with open(log_folder + log_file, 'w+') as f:
        json.dump(log, f, indent=4)

    # Return game states for each run
    game_states = [game for (game, log) in results]
    return game_states





#################
### Execution ###
#################

def parse_args():
    args = argparse.ArgumentParser()

    args.add_argument("--set", type=str, choices=["mini", "train", "validation", "test"], default="mini")
    args.add_argument("--num_agents", type=int, default=5)
    args.add_argument("--backtrack", type=float, default=0.6)
    args.add_argument("--num_steps", type=int, default=10)
    args.add_argument("--resampling", type=str, choices=["linear", "logistic", "max", "percentile", "linear_filtered", "max_unique"], default="linear")
    args.add_argument("--k", type=int, default=1)
    args.add_argument('--debugging', type=int, default=0)
    args.add_argument('--pruning', type=int, default=1)
    args.add_argument('--repeats', type=int, default=1)
    args.add_argument('--caching', type=int, default=1)
    args.add_argument('--batching', type=int, default=1)
    args = args.parse_args()
    return args

async def main():
    args = parse_args()

    # Select parameters
    set = args.set                                                  # Set of data to be used
    num_agents = args.num_agents                                    # Number of agents
    k = args.k                                                      # Resampling every <k> steps
    backtrack = args.backtrack                                      # Backtrack decaying coefficient
    origin_value = 20 * 3 / backtrack if backtrack !=0 else 20 * 3  # The evaluation of the origin
    num_steps = args.num_steps                                      # Max allowed steps
    resampling_method = args.resampling                             # Resampling method
    debugging = args.debugging                                      # Number of puzzles to run
    pruning = args.pruning                                          # Whether to prune or not
    repeats = args.repeats                                          # Number of times to repeat the experiment
    caching = args.caching                                          # Whether to cache the evaluations or not
    batching  = args.batching                                       # Whether to use batching or not

    log_file_ = f"{set}_{num_agents}ag_{num_steps}st_{k}k_{origin_value}or_{backtrack}b_{resampling_method}.json"

    foa_options = {
        "num_agents": num_agents,
        "k": k,
        "origin_value": origin_value,
        "num_steps": num_steps,
        "backtrack": backtrack,
        "resampling_method": resampling_method,
        "pruning": pruning,
        "caching": caching,
        "batching": batching
    }


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

        run_options = {
            "task": "GameOf24",
            "set":set,
            "seed":seed,
            "debugging":debugging
        }

        # Run
        results = await run(run_options, foa_options, log_file=log_file)

        # Total accuracy and cost computation
        n_success = 0
        for game in results:
            verifications = [GameOf24Agent.verify(result) for result in game]
            if {"r": 1} in verifications:
                n_success += 1
        accuracy = n_success * 100 / len(results)
        print(f"Accuracy : {accuracy:.2f}\n")
        print(f"File name : {log_file}\n\n\n\n\n")

        #Update actual cost.
        update_actual_cost(api)

        # Empty api cost so multiple repeats are not cumulative
        api.empty_tabs()




if __name__ == "__main__":
    asyncio.run(main())