import os, argparse
from src.methods.agents import Agents
from src.tasks.game24 import Game24
from src.models import OpenAIBot
from src.utils import delete_file


def run(args):
    task_start_index = args.task_start_index
    task_end_index = args.task_end_index
    n_evaluations = args.n_evaluations
    n_agents = args.n_agents
    model_name = args.model_name
    init = args.init

    # Log file initialization (Choose name + delete if it already exists)
    file_name=f"{model_name}__{n_agents}agents_{n_evaluations}evaluations_{task_start_index}start_{task_end_index}end.json"
    current_path = os.getcwd()
    log_path = os.path.join(current_path, "logs")
    file_path = os.path.join(log_path, file_name)
    delete_file(file_path)


    # Create agents
    ## TODO: Add model + task selection
    bot = OpenAIBot(model=model_name)
    for idx_input in range(task_start_index, task_end_index):
        agents = Agents(task=Game24, idx_input=idx_input, n_agents=n_agents, model=bot, init=init)

        # Run agents
        for i in range(agents.max_steps-int(init)):
            agents.step()
            agents.evaluate(n=n_evaluations)
            if i!=agents.max_steps-1:       # Resampling condition (when to resample)
                agents.resample()
        agents.test_output()

        # Log results
        agents.create_log(repo_path=log_path, file_name=file_name)
    
    Game24.get_accuracy(file_path)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--init", type=bool, default=True)
    args.add_argument("--task_start_index", type=int, default=900)
    args.add_argument("--task_end_index", type=int, default=910)
    args.add_argument("--n_evaluations", type=int, default=3)
    args.add_argument("--n_agents", type=int, default=5)
    args.add_argument("--model_name", type=str, choices=["gpt-4", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"] , default="gpt-3.5-turbo-1106")
    
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)