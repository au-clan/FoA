import os
import json
import time
import sys
import io
import asyncio
import argparse
sys.path.append(os.getcwd()) # Project root!!
from src.agents.rafaagent import TreeOfThoughtAgent
from src.states.rafaenv import Game24
from src.agents import gpt_usage

def get_model():
    gpt_model = "gpt-4.1-nano-2025-04-14"
    llama_model = "llama-3.3-70b-versatile"
    return gpt_model

async def run(args):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    #model = get_model()
    model = args.backend
    print("reflection type: ", args.method_reflexion_type)
    feedback = False if args.feedback == 'False' else True
    reflect = False if args.reflect == 'False' else True
    feedback_string = False if args.feedback_string == 'False' else True
    

    agent = TreeOfThoughtAgent(
        backend=model, temperature=0.7, prompt_sample="standard",
        method_generate=args.method_generate, method_evaluate="value",
        method_select="greedy", method_reflexion_type=args.method_reflexion_type,
        n_generate_sample=10, n_evaluate_sample=1, n_select_sample=1,
        k = args.k, lower_limit = args.lower_limit, upper_limit = args.upper_limit,
        feedback=feedback
    )
    env = Game24(datadir=f'24_tot.csv', feedback=feedback, max_steps=20, split=args.split, reflect=reflect, feedback_string=feedback_string)
    cur_time = int(time.time())
    num_puzzles = 30 if args.split == "uniform-validation" else 60
    file = f'logs/recent/gameof24/RAFA/game24/{agent.backend}_{args.method_reflexion_type}_k_{args.k}_limit_{args.upper_limit}_summary_{cur_time}.json'

    os.makedirs(os.path.dirname(file), exist_ok=True)
    logs = []

    # Example: range(0, 100) to run 100 puzzles
    puzzle_tasks = [
        asyncio.create_task(
            run_puzzle(
                i, env, agent, logs, file
            )
        )
        for i in range(0, num_puzzles)
    ]
    await asyncio.gather(*puzzle_tasks)
    

async def run_puzzle(i, env, agent, logs, file):
    obs = env.reset(i)
    log = {'idx': i, 'agent_info': [], 'env_info': []}
    done = False
    total_reward = 0
    success = False
    j = 0

    while not done:
        j += 1
        print(f"Iteration {j}")

        # Step: agent decides action
        action, agent_info = agent.act(env, obs)
        #print("Action:", action)
        print("Action: ", action)
        # Step: environment reacts
        obs, reward, done, env_info = env.step(action)

        total_reward += reward
        if reward >= 10:
            success = True

        # Step: update agent
        agent.update(obs, reward, done, env_info)

        # Step: record logs for this iteration
        log['agent_info'].append(agent_info)

        # Get usage and attach it to this env_info step
        usage = gpt_usage(agent.backend)
        env_info['usage_so_far'] = usage
        log['env_info'].append(env_info)

        # Save intermediate state (ensures file is non-empty and debuggable)
        with open(file, 'w') as f:
            json.dump(logs + [log], f, indent=4)

    # Finalize log
    log['total_reward'] = total_reward
    log['success'] = f'success: {success}'
    logs.append(log)

    # Final write
    with open(file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Finished puzzle {i} | Success: {success} | Total reward: {total_reward}")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str,
                      choices=['gpt-4.1-nano-2025-04-14', 'llama-3.3-70b-versatile'],
                      default='gpt-4.1-nano-2025-04-14')


    args.add_argument('--method_reflexion_type', type=str,
                      choices=['list', 'k_most_recent', 'summary'],
                      default='list') 
    args.add_argument('--method_generate', type=str,
                      choices=['propose', 'single'],
                      default='propose')  
    args.add_argument('--split', type=str,
                      choices=['uniform-validation', 'uniform-test'],
                      default='uniform-validation')                                   
    args.add_argument('--k', type=int, default=3)
    args.add_argument('--lower_limit', type=int, default=2)
    args.add_argument('--upper_limit', type=float, default=3)
    args.add_argument('--feedback', type=str, 
                      choices=['False', 'True'],
                      default='True')
    args.add_argument('--reflect', type=str, 
                      choices=['False', 'True'],
                      default='True')
    args.add_argument('--feedback_string', type=str, 
                      choices=['False', 'True'],
                      default='True')
    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    asyncio.run(run(args))