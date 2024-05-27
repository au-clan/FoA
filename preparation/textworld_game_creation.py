import os
import textworld
import textworld.gym

# this must be run from the root of the project
assert os.getcwd().split("/")[-1] == "FoA", "Please run the script from the root directory of the project."

# Define the scenarios
scenarios = ["tw-simple", "tw-coin_collector", "tw-cooking", "tw-treasure_hunter"]

# configs ={
#     "tw-simple --rewards balanced --goal brief": 10,
#     "tw-coin_collector --level 1": 5,
#     "tw-cooking": 5,
#     "tw-treasure_hunter --level 5": 5,
# }

configs ={
    "tw-treasure_hunter --level 15": 5,
}

gamefiles = []

# Generate the games
for scenario, num_games in configs.items():
    for i in range(num_games):
        # Use the loop index as the seed
        seed = i

        game_name = f"{scenario}_{seed}".replace(" ", "_")

        # Define the game file name
        game_file = f"data/datasets/tw_games/{game_name}.ulx"
        gamefiles.append(game_file)

        # skip if exists
        if os.path.exists(game_file):
            continue

        # Generate the game
        os.system(f"tw-make {scenario} --output {game_file} -f -v --seed {seed}")