import os, argparse
from src.methods.agents import Agents
from src.tasks.game24 import Game24
from src.models import OpenAIBot


def run(args):
    idx_input = args.idx_input
    n_evaluations = args.n_evaluations
    n_agents = args.n_agents
    model_name = args.model_name

    # Create agents
    ## TODO: Add model + task selection
    bot = OpenAIBot(model=model_name)
    agents = Agents(task=Game24, idx_input=idx_input, n_agents=n_agents, model=bot)

    # Run agents
    for i in range(agents.max_steps):
        agents.step()
        agents.evaluate(n=n_evaluations)
        if i!=agents.max_steps-1:       # Resampling condition (when to resample)
            agents.resample()

    # Log results
    current_path = os.getcwd()
    log_path = os.path.join(current_path, "logs")
    agents.create_log(repo_path=log_path)

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--idx_input", type=int, default=8)
    args.add_argument("--n_evaluations", type=int, default=3)
    args.add_argument("--n_agents", type=int, default=5)
    args.add_argument("--model_name", type=str, choices=["gpt-4", "gpt-3.5-turbo-1106", "gpt-3.5-turbo-0125"] , default="gpt-3.5-turbo-0125")
    
    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(args)
    run(args)