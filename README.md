# FoA
Master thesis project on Fleet of Agents runtime.

## Setup
1. Clone this repository, eg.
```bash
git clone https://github.com/epfl-dlab/FoA.git
cd FoA
```
2. Set up an environment variable for OpenAI API, eg.
```bash
conda create -n foa python=3.11
conda activate foa
conda env config vars set OPENAI_API_KEY=<YOUR_API_KEY>
conda activate foa
```
3. Install requirements, eg.
```bash
pip install -r requirements.txt
```

## Quickstart
To run an experiment you can simply use:
```bash
python run.py
```
Arguments
- `--difficulty`: Selects the difficulty of the samples going from 0 (easiest) to 9 (samples used for the ToT paper).
- `--n_samples`: Number of samples/games to run during the experiment.
- `--n_evaluations`: Number of times to evaluate each state in order to give it a final value (=sum of all evaluations).
- `--n_agents`: Number of agents to use for the experiment.
- `--back_coef`: Backtracking coefficient (min:0 -> No backtracking allowed, max:1 -> Old states value does not depreciate).
- `--max_steps`: Max steps agents are allowed to execute.
- `--model_name`: Name of OpenAI model to use.
- `--init` (flag): Whether to use initialization or not (default=False). Using initialization guarantees that the first step of each agent is unique but random.
- `--foa_prompt` (flag): Whether to use foa specific prompt or prompt from ToT paper (default=False -> ToT).

Default arguments
```bash
python run.py  --difficulty 0 --n_samples 50 --n_evaluations 3 --n_agents 5 --back_coef 0.8 --max_steps 10 --model_name gpt-3.5-turbo-1106
```
