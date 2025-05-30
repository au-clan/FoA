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

    agent = TreeOfThoughtAgent(
        backend=model, temperature=0.7, prompt_sample="standard",
        method_generate="propose", method_evaluate="value",
        method_select="greedy", method_reflexion_type=args.method_reflexion_type,
        n_generate_sample=10, n_evaluate_sample=1, n_select_sample=1,
        k = args.k, limit = args.limit
    )
    env = Game24(f'24_tot.csv', True, 8)
    cur_time = int(time.time())
    file = f'logs/recent/gameof24/RAFA/game24/{agent.backend}_0.7_{agent.method_generate}_{agent.n_generate_sample}_{agent.method_evaluate}_{agent.n_evaluate_sample}_{agent.method_select}_{agent.n_select_sample}_60PuzzlesNoSelectionState_time{cur_time}.json'

    os.makedirs(os.path.dirname(file), exist_ok=True)
    logs = []

    # Example: range(0, 100) to run 100 puzzles
    puzzle_tasks = [
        asyncio.create_task(
            run_puzzle(
                i, env, agent, logs, file
            )
        )
        for i in range(0, 1)
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
        print("Iteration: ", j)
        action, agent_info = agent.act(env, obs)
        print("List of actions after agent.act: ", action)
        print("Agent info after agent.act: ", agent_info)
        obs, reward, done, env_info = env.step(action)
        print("env info after env.step: ", env_info)

        total_reward += reward
        if reward >= 10:
            success = True

        agent.update(obs, reward, done, env_info)
        log['agent_info'].append(agent_info)
        log['env_info'].append(env_info)
        log['usage_so_far'] = gpt_usage("gpt-4.1-nano-2025-04-14")
        with open(file, 'w') as f:
            json.dump(logs, [log], f, indent=4)

    log['total_reward'] = total_reward
    log['success'] = f'success: {success}'
    #Hej

    logs.append(log)
    #Hej
    with open(file, 'w') as f:
        json.dump(logs, f, indent=4)

    print(f"Finished puzzle {i} | Success: {success} | Total reward: {total_reward}")

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--backend', type=str,
                      choices=['gpt-4.1-nano-2025-04-14', 'gpt-4.1-nano-2025-04-14'],
                      default='gpt-4.1-nano-2025-04-14')


    args.add_argument('--method_reflexion_type', type=str,
                      choices=['list', 'k-most-recent', 'summary'],
                      default='list') 
    args.add_argument('--k', type=int, default=3)
    args.add_argument('--limit', type=int, default=15)

    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)
    asyncio.run(run(args))