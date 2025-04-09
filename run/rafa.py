import os
import json
import time
import sys
import io
sys.path.append(os.getcwd()) # Project root!!
from src.agents.rafaagent import TreeOfThoughtAgent
from src.states.rafaenv import Game24
from src.agents import gpt_usage


def run():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    agent = TreeOfThoughtAgent(backend="llama-3.3-70b-versatile", temperature=0.7, prompt_sample="standard",
                      method_generate="propose", method_evaluate="value",
                      method_select="greedy", n_generate_sample=10,
                      n_evaluate_sample=1, n_select_sample=1)
    env = Game24(f'24_tot.csv', True, 20)
    cur_time = int(time.time())
    file = f'logs/recent/gameof24/RAFA/game24/llama-3.3-70b-versatile_0.7_propose_10_value_1_greedy_1_time{cur_time}.json'
    
    os.makedirs(os.path.dirname(file), exist_ok=True)
    logs = []
    # for i in range(args.task_start_index, args.task_end_index):
    for i in range(1-1, 0-1, -1):
        obs = env.reset(i)
        log = {'idx': i, 'agent_info': [], 'env_info': []}
        done = False
        j = 0
        while not done:
            j +=1
            print("Iteration: ", j)
            action, agent_info = agent.act(env, obs)
            print("List of actions after agent.act: ", action)
            print("Agent info after agent.act: ", agent_info)
            obs, reward, done, env_info = env.step(action)
            print("env info after env.step: ", env_info)
            agent.update(obs, reward, done, env_info)
            log['agent_info'].append(agent_info)
            log['env_info'].append(env_info)
            #print(obs)
            #print(reward, done, env_info)
            log['usage_so_far'] = gpt_usage("llama-3.3-70b-versatile")
            tmp_logs = logs + [log]
            with open(file, 'w') as f:
                json.dump(tmp_logs, f, indent=4)
        logs.append(log)
        with open(file, 'w') as f:
            json.dump(logs, f, indent=4)


if __name__ == '__main__':
    run()