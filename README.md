# Official Repository of Fleet of Agents

![Foa detailed method gif](/pics/foa_detailed.gif)

This is the official implementation of Fleet of Agents: Coordinated Problem Solving with Large Language Models

In this paper, we introduce Fleet of Agents (FoA), a novel and intuitive yet principled framework utilizing LLMs as agents to navigate through dynamic tree searches, employing a genetic-type particle filtering approach. FoA spawns a multitude of agents, each exploring the search space autonomously, followed by a selection phase where resampling based on a heuristic value function optimizes the balance between exploration and exploitation. This mechanism enables dynamic branching, adapting the exploration strategy based on discovered solutions.

## Setup
1. Set up a conda environment. TogetherAI or Groq API keys should be set as well if you'd like to test open-source models.
```bash
conda create -n foa python=3.12.3
pip install -r requirements.txt
conda env config vars set OPENAI_API_KEY=<...>
conda env config vars set GROQ_API_KEY=<...> # (OPTIONAL)
conda env config vars set TOGETHER_API_KEY=<...> # (OPTIONAL)
```

2. Please set up your WebShop environment following the instructions of its original repository: [WebShop](https://github.com/princeton-nlp/WebShop). In case the WebShop URL needs to be updated, you can do so in our `utils.py` file (line 122).

## Paper Results
To replicate the results reported in our paper (Section 4. Experiments) Feel free to execute any of the following scripts.

```bash
sh run/gameof24.sh
sh run/crosswords.sh
sh run/webshop.sh
```

## Additional Experiments
To perform additional experiments, such as the ablation studies presented within the paper, feel free to add any of the following arguments to the scripts presented above.

- ``--caching 0``: Removes the caching mechanism.
- ``--batching 0``: Removes the batching mechanism.
- ``--repeats X``: Repeats the experiment X times.

To change the configurations of FoA you can use the following.
- ``--num_agents X``: Sets the number of agents deployed.
- ``--num_steps X``: Sets the max number of steps each agent is allowed to take.
- ``--backtrack X``: Sets the discount factor $X \in [0, 1]$.
- ``k X`` : Sets the resampling frequency.
- ``--resampling Y``: Sets the resampling strategy. Current iteration supports ``linear``, ``linear_filtered``, ``max``, ``max_unique`` and ``percentile``.


## Citations
Your support would be greatly appreciated if you find FoA interesting or useful. Please acknowledge our work by citing the paper and giving this repository a star. Feel free to open an issue if you have any questions.


```bibtex
@inproceedings{
klein2025fleet,
title={Fleet of Agents: Coordinated Problem Solving with Large Language Models},
author={Lars Henning Klein and Nearchos Potamitis and Roland Aydin and Robert West and Caglar Gulcehre and Akhil Arora},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=yNpYb376zf}
}
```