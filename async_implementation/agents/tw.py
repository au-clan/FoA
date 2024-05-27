import textworld
import random
from uuid import uuid4

class TextWorldAgent:

    def __init__(self, env_id, random_seed, replay_actions=[]):

        self.unique_id = uuid4()
        self.random_seed = random_seed
        self.observations = []
        self.rewards = []
        self.infos = []
        self.action_history = []

        self.env_id = env_id
        self.env = textworld.gym.make(env_id)
        obs, infos = self.env.reset()
        obs = self.strip_obs(obs)
        self.observations.append(obs)
        self.infos.append(infos)
        
        assert "max_score" in infos, "The environment must provide the max_score from EnvInfos"
        self.max_score = infos["max_score"]

        self.terminal = False

        for action in replay_actions:
            obs, reward, terminal, infos = self.env.step(action)
            self.terminal = terminal
            self.observations.append(obs.strip())
            self.rewards.append(reward)
            self.infos.append(infos)
            self.action_history.append(action)
            if terminal:
                print(f"Agent terminated while cloning!")

    def reset(self):
        self.observations = []
        self.rewards = []
        self.infos = []
        self.action_history = []

        obs, infos = self.env.reset()
        obs = self.strip_obs(obs)
        self.observations.append(obs)
        self.infos.append(infos)

        self.terminal = False

    def strip_obs(self, obs):
        return obs.replace(r"""


                    ________  ________  __    __  ________        
                   |        \|        \|  \  |  \|        \       
                    \$$$$$$$$| $$$$$$$$| $$  | $$ \$$$$$$$$       
                      | $$   | $$__     \$$\/  $$   | $$          
                      | $$   | $$  \     >$$  $$    | $$          
                      | $$   | $$$$$    /  $$$$\    | $$          
                      | $$   | $$_____ |  $$ \$$\   | $$          
                      | $$   | $$     \| $$  | $$   | $$          
                       \$$    \$$$$$$$$ \$$   \$$    \$$          
              __       __   ______   _______   __        _______  
             |  \  _  |  \ /      \ |       \ |  \      |       \ 
             | $$ / \ | $$|  $$$$$$\| $$$$$$$\| $$      | $$$$$$$\
             | $$/  $\| $$| $$  | $$| $$__| $$| $$      | $$  | $$
             | $$  $$$\ $$| $$  | $$| $$    $$| $$      | $$  | $$
             | $$ $$\$$\$$| $$  | $$| $$$$$$$\| $$      | $$  | $$
             | $$$$  \$$$$| $$__/ $$| $$  | $$| $$_____ | $$__/ $$
             | $$$    \$$$ \$$    $$| $$  | $$| $$     \| $$    $$
              \$$      \$$  \$$$$$$  \$$   \$$ \$$$$$$$$ \$$$$$$$ 

""", "").strip()

    def format_user_message(self, obs, infos):
        """converts an observation from the environment into a user message which is designed to prompt for the next action from the assistant"""
        return {
            "role": "user",
            "content": obs
        }

    def format_first_user_message(self, obs, infos):
        """converts an observation from the environment into a user message which is designed to prompt for the next action from the assistant"""

        scenario = "simple"
        scenario = "cooking"
        if scenario == "simple":
            content = f"""This is the beginning of your text-based adventure. Here is the first message from the game, it will set the stage and give you a task that you're supposed to fulfil to win the game (the task may only be hinted at): {obs} + "\n\nWhat do you want to do next? Remember, your answers will be put directly into the CLI, so you must stick to the given input format without any additional comments."""
        elif scenario == "cooking":
            content = f"""This is the beginning of a text-based cooking game. The game will give you instructions on what to do, typically you will have to read a cookbook, gather ingredients, process ingredients in some way, prepare a meal (this must only be done after you've completed all prerequisite tasks) and finally eat the meal (this ends the game, if you have followed all instructions from the cookbook, you will win). Here is the first message from the game: {obs} + "\n\nWhat do you want to do next? Remember, your answers will be put directly into the CLI, so you must stick to the given input format without any additional comments."""
        else:
            assert False

        return {
            "role": "user",
            "content": content
        }

    def format_last_user_message(self, obs, infos):
        """converts an observation from the environment into a user message which is designed to prompt for the next action from the assistant"""
        return {
            "role": "user",
            "content": obs + "\n\nWhat do you want to do next? Remember, your answers will be put directly into the CLI, so you must stick to the given input format without any additional comments. Admissible answers are: " + ", ".join(
                infos["admissible_commands"])
        }

    def format_assistant_message(self, action):
        """converts an action into an assistant message which is designed to prompt for the next observation from the environment"""
        return {
            "role": "assistant",
            "content": action
        }

    def format_system_message(self, obs, infos):
        """converts an action into a system message which is designed to prompt for the next observation from the environment"""
        verbs = ", ".join(["'" + v + "'" for v in infos["verbs"]])
        command_templates = ", ".join(["'" + v + "'" for v in infos["command_templates"]])
        return {
            "role": "system",
            "content": f"""
You're playing a text-based command line adventure game.
Your answers will be put directly into the CLI, so you must stick to the given input format without any additional comments.
The following verbs are available: {verbs}. You may combine them into actions like 'go north' or 'take key'. The following action templates are available: {command_templates}"""
        }

    async def step(self, api, namespace):
        assert len(self.observations) == len(self.infos)
        assert len(self.observations) == len(self.action_history) + 1
        assert len(self.observations) > 0, "resetting the environment must generate one observation at the beginning"

        message_history = []
        # initialize with a system message, which may look at the very first observation and info
        message_history.append(self.format_system_message(self.observations[0], self.infos[0]))

        for i in range(len(self.observations)):
            # we have special formatting for the first and last user message
            if i == 0:
                message_history.append(self.format_first_user_message(self.observations[i], self.infos[i]))
            elif i + 1 == len(self.observations):
                message_history.append(self.format_last_user_message(self.observations[i], self.infos[i]))
            else:
                message_history.append(self.format_user_message(self.observations[i], self.infos[i]))

            # is there an accompanying action?
            if i < len(self.action_history):
                message_history.append(self.format_assistant_message(self.action_history[i]))

        
        # get next action from the system
        response = await api.buffered_request(message_history, key=self.hash(), namespace=namespace)
        message = response

        # ToDo: parse action from response, what do we do if the action is invalid?
        # for now we check for any string match and if none is found, we take a random action
        admissible_actions = self.infos[-1]["admissible_commands"]
        chosen_action = None
        for action in admissible_actions:
            if action in message:
                chosen_action = action
                break

        if chosen_action is None:
            random.seed(self.random_seed)
            chosen_action = random.choice(admissible_actions)

        #print(f"Chosen action: {chosen_action}")

        # apply the action to the environment
        obs, reward, terminal, infos = self.env.step(chosen_action)
        self.terminal = terminal
        obs = self.strip_obs(obs)
        self.observations.append(obs)
        self.rewards.append(reward)
        self.infos.append(infos)
        self.action_history.append(chosen_action)

    async def clone(self, random_seed=None):
        if random_seed is None:
            random_seed = self.random_seed
        cloned_agent = TextWorldAgent(self.env_id, random_seed, self.action_history)
        assert cloned_agent.observations == self.observations, "cloned agent should have the same observations as the original agent"
        assert cloned_agent.infos == self.infos, "cloned agent should have the same infos as the original agent"
        assert cloned_agent.action_history == self.action_history, "cloned agent should have the same action history as the original agent"
        if cloned_agent.terminal:
            print(f"Original Agent terminated : {self.terminal}")
            print(f"Cloned Agent terminated : {cloned_agent.terminal}")
            print(f"Original Agent actions : {self.action_history}")
            print(f"Cloned Agent actions : {cloned_agent.action_history}")
        assert not cloned_agent.terminal, "it doesn't make sense to clone a terminal agent, this points to a logic error in the outer algorithm"
        return cloned_agent
    
    def hash(self):
        return hash((self.env_id, " ".join(self.observations), " -> ".join(self.action_history), self.random_seed))
    
    def has_won(self):
        if self.rewards[-1] == self.max_score:
            # Just making sure
            assert self.infos[-1]["won"], "if the score is max, the game must be won" 
        return self.rewards[-1] == self.max_score