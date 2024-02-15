import os, argparse
from datetime import datetime

# Custom
from src.methods.agents import Agents
from src.tasks.game24 import Game24
from src.models import OpenAIBot
from src.utils import delete_file, create_folder


def run(args):
    task_start_index = args.difficulty * 100
    task_end_index = args.difficulty * 100 + args.n_samples
    n_evaluations = args.n_evaluations
    n_agents = args.n_agents
    model_name = args.model_name
    init = args.init
    foa_prompt = args.foa_prompt
    back_coeff = args.back_coeff
    max_steps = args.max_steps

    # Log file initialization (Choose name + delete if it already exists)
    file_name=f"{model_name}__{n_agents}agents_{str(foa_prompt)[0]}foa_{n_evaluations}evaluations_{task_start_index}start_{task_end_index}end.json"
    current_path = os.getcwd()
    log_path = os.path.join(current_path, f"logs/{datetime.now().date()}")
    create_folder(log_path)
    file_path = os.path.join(log_path, file_name)
    delete_file(file_path)


    # Create agents
    ## TODO: Add model + task selection
    bot = OpenAIBot(model=model_name)
    for idx_input in range(task_start_index, task_end_index):
        agents = Agents(task=Game24, idx_input=idx_input, n_agents=n_agents, init=init, back_coef=back_coeff, model=bot,  foa_prompt=foa_prompt)

        # Run agents
        for i in range(max_steps-int(init)):
            agents.step()
            done, _ = agents.test_output()
            if done:
                break
            agents.evaluate(n=n_evaluations)
            if i!=max_steps-int(init)-1:    # Resampling condition (when to resample)
                agents.resample()
        

        # Log results
        agents.create_log(repo_path=log_path, file_name=file_name)
    
    Game24.get_accuracy(file_path)
    print(f"Logs saved in : \n\t'{file_path}'")
def parse_args():
    args = argparse.ArgumentParser()
    
    args.add_argument("--difficulty", type=int, choices=list(range(10)), default=0)
    args.add_argument("--n_samples", type=int, default=50)
    args.add_argument("--n_evaluations", type=int, default=3)
    args.add_argument("--n_agents", type=int, default=5)
    args.add_argument("--back_coef", type=float, default=0.8)
    args.add_argument("--max_steps", type=int, default=10)
    args.add_argument("--model_name", type=str, choices=["gpt-4", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"] , default="gpt-3.5-turbo-1106")
    args.add_argument("--init", action="store_true")
    args.add_argument("--foa_prompt", action="store_true")
    
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)