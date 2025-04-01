import os
import pathlib
import sympy
import pandas as pd
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task24 import MCTS_Task
import argparse
from datetime import datetime
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *
from utils.self_consistency import get_consistency_output_scibench
from models.model import gpt_usage
from openai import OpenAI

print("Cleaning the log file...")   
with open('zzz.log', 'w') as f:
    pass 

def run(arguments):
    time = datetime.now()
    print('-'*30, 'Begin testing', '-'*30, '\n')
    file = "data/dataset_game24.csv.gz"
    try:
        df = pd.read_csv(file, compression='gzip')
        start = 900
        end = 1000
        data_list = df['Puzzles'].tolist()[start:end]
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    #assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"
    
    client=OpenAI()

    output_list = []
    correct_count = 0
    for i in range(data_len):
        # solve
        print(f'Begin to solve the problem {i+1}...\n')
        data = data_list[i]
        if arguments.mode == 'cot':
            assert False
            # Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature, evaluate=arguments.evaluate)
            # if arguments.consistency:
            #     outputs = []
            #     for cnt in range(3):
            #         output = Task.run()
            #         outputs.append(output)
            #     output = get_consistency_output_scibench(outputs)
            # else:
            #     output = Task.run()

        elif arguments.mode == 'tot':
            assert False
            # Task = ToT_Task(data, arguments.propose_method, arguments.value_method, arguments.algorithm,
            #                 arguments.branch, arguments.select_branch, arguments.max_depth, arguments.end_gate,
            #                 arguments.select_method, arguments.temperature, use_case_prompt=arguments.use_case_prompt,
            #                 low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            # output, root = Task.run()
            # if arguments.visualize:
            #     visualize(root, Task, arguments.task_name, arguments.file, i + 1)
        else:
            Task = MCTS_Task(data, client, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            print(f"Running MCTS")
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)

        # evaluate metrics
        if arguments.evaluate:

            #result = verify_float(answer, output['summary'])
            result = test_output24(data, output['solution'])
            print(f'{output=}')
            print(f'{result=}')
            output.update({'answer': "", 'accurate': result})
            if result:
                print(f'The answer of problem {i+1} is correct.\n')
                correct_count += 1
            else:
                print(f'The answer of problem {i+1} is wrong.\n')
        print(f'The solution to problem {i+1} is complete.\n')
        cost = gpt_usage()
        output["cost"] = cost
        print(f'Cost: {cost}\n')

        # output
        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/outputs/{start}_{end}/{Task.mode}')
        output_file = f'{base_dir}/outputs/{start}_{end}/{Task.mode}/{Task.propose_method}_{Task.value_method}_{time.hour}_{time.minute}_{arguments.branch}b_{arguments.iteration_limit}T.json'
        output_list.append(output)
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)

    print('_' * 60)
    # accuracy
    if args.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='thermo_standardized')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local'], default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='tot')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=100)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)  # End threshold
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=str, default='scibench')  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_case_prompt', type=bool, default=False)  # Use sample prompts
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--consistency', type=bool, default=False)

    arguments = base_args.parse_args()
    return arguments




def test_output24(x, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', x)
        if sorted(numbers) != sorted(problem_numbers):
            out = {'r': 0}
        try:
            # print(sympy.simplify(expression))
            out = {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            out = {'r': 0}
        return out == {'r': 1}

if __name__ == '__main__':
    args = parse_args()
    run(args)