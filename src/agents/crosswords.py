import re
import json
import asyncio
import random
from typing import Tuple

from numpy import mean
from src.prompts.adapt import crosswords as adapt_prompts
from src.prompts.totor import crosswords as totor_prompts
from src.states.crosswords import CrosswordsState


class CrosswordsAgent:

    @staticmethod
    async def get_candidates(state: CrosswordsState, api, namespace, candidate_cache, n:int =2, caching=True)-> dict:
        """
        Given a state, return a dictionary of candidate actions along with its scores.
        """
        # Render the state
        obs = CrosswordsState.render(state)

        # Return cached candidates if they exist
        if obs in candidate_cache and caching:
            return candidate_cache[obs]

        # Prepare the prompt
        if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
            prompt = adapt_prompts.propose_prompt.format(input=obs)
        else:
            prompt = totor_prompts.propose_prompt.format(input=obs)

        # Get candidate actions and their scores
        coroutines = []
        for _ in range(n):
            coroutines.append(api.buffered_request(prompt, key=hash(state), namespace=namespace))
        responses = await asyncio.gather(*coroutines)

        # Parse the responses and add the scores for each action
        candidates_to_scores = {}
    
        for response in responses:
            # Azure-Filtered reponse
            if response == None:
                continue
            parsed_response = parse_response(response)
            # Example: response = {'h1. apple': "certain", 'h2. banana': "high", 'h3. apple': "medium", 'h4. apple': "low", 'h5. apple': "low"}
            # Example: mapping = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}
            # Example: parsed_response = {"h1. apple": 1, "h2. banana": 0.5, "h3. apple": 0.2, "h4. apple": 0.1, "h5. apple": 0.1}
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        filtered_candidate_to_score = {k: v for k, v in candidates_to_scores.items()  if provokes_change(state, k)}

        #Update cache
        if len(filtered_candidate_to_score) > 0:
            candidate_cache[obs] = filtered_candidate_to_score

        # Example: {"h1. apple": 2.5, "h2. banana": 1.0, "h3. apple": 0.5, "h4. apple": 0.4, "h5. apple": 0.3}
        return filtered_candidate_to_score
    
    @staticmethod
    async def step(state: CrosswordsState, api, namespace, candidate_cache: dict, caching:bool = True)-> Tuple[CrosswordsState, str]:
        """
        Given a state, returns the next state one.
        """
        # Get next step suggestions/actions and pick one of the ones with the highest value
        
        # suggestions = {"h1. apple": 2.5, "h2. banana": 1.0, "h3. apple": 0.5, "h4. apple": 0.4, "h5. apple": 0.3}
        suggestions = await CrosswordsAgent.get_candidates(state, api, candidate_cache=candidate_cache, namespace=namespace, caching=caching)
        suggestions = dict(sorted(suggestions.items(), key=lambda item: item[1], reverse=True))


        # Pick the best allowed (by ToT) mutation
        mutation_found = False
        for action, _ in suggestions.items():
            
            # Get the position and word from the action
            parsed_action = parse_action(action)
            pos, word = parsed_action

            # Assert the action is valid TODO: Maybe change to actual assert (?)
            if len(parsed_action) != 2:
                return 'Invalid! Format should be like "h1. apple"', 0, False, {}
            if len(word) != 5:
                return 'Invalid! Word should have 5 letters.', 0, False, {}
            
            # Compute new board by applying the action
            new_board = state.board.copy()
            if pos.startswith('h'):
                idx = int(pos[1:]) - 1
                new_board[idx*5:(idx+1)*5] = list(word.upper())
            elif pos.startswith('v'):
                idx = int(pos[1:]) - 1
                new_board[idx::5] = list(word.upper())
                idx += 5  # for later status update
            else:
                return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
            
            new_ans = CrosswordsState.get_ans(new_board)
            new_status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(state.status, state.ans, new_ans)]
            new_status[idx] = 1

            if any(status == 2 for status in new_status):
                # As per ToT: If the action changes a different word, discard it.
                continue
            else:
                mutation_found = True
                break
        
        if mutation_found:
            # Return the next mutation
            random.seed(state.randomness)
            next_state = CrosswordsState(
                data=state.data,
                board_gt=state.board_gt,
                ans_gt=state.ans_gt,
                board=new_board, 
                ans=new_ans, 
                status=new_status,
                steps = state.steps + [action],
                randomness=random.randint(0, 1000)
                )
            return next_state
    
        else:
            """
            TODO: Cleaner solution.
            If this no "mutation" is found the ToT backtracks. In our case backtrack is replaced by resampling.
            This is quite a hacky solution. Basically returns the same state with all answers filled with "PRUNE".
            This is a signal for our algorithm to prune the state.
            """
            next_state = CrosswordsState(
            data=state.data,
            board_gt=state.board_gt,
            ans_gt=state.ans_gt,
            board=state.board, 
            ans=["PRUNE"]*10, 
            status=state.status,
            steps=state.steps,
            randomness=state.randomness
            )
            return next_state
        
    @staticmethod
    async def evaluate(state, api, namespace, value_cache, caching: bool=True):
        """
        Evaluates the current state and returns a value number.
        The state is evaluated line by line.
        """

        # Count of the results for each line
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        
        for ans, data in zip(state.ans, state.data):

            # Skip answers with 4 or 5 missing letters
            if ans.count('_') >= 4:
                continue
            
            # Parse the answer along with the original question
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            
            # Prepare the prompt
            if any(author in api.model for author in ["meta", "google", "mistral", "gpt-4o"]):
                prompt = adapt_prompts.value_prompt.format(input=line)
            else:
                prompt = totor_prompts.value_prompt.format(input=line)
            
            # Get a value for the line from the set {sure, maybe, impossible} 
            if prompt in value_cache and caching:
                response = value_cache[prompt]
            else:
                response = await api.buffered_request(prompt, key=hash(state), namespace=namespace)
                value_cache[prompt] = response

            # Azure-Filtered response
            if response is None:
                continue
            
            # Parse the response and update the count
            parsed_response = response.split('\n')[-1].strip()
            if parsed_response in count:
                count[parsed_response] += 1

        # Map the count to a value number 
        value_map = {'impossible': -20, 'maybe': 5, 'sure': 20} #TODO: ad hoc
        value_number  = sum(value * value_map[name] for name, value in count.items())
        return max(value_number, 0)
    
    @staticmethod
    def verify(state: CrosswordsState)->dict:
        """
        Verifies the output of a given task
            1. Checks if the numbers used are the same as the ones provided.
            2. Checks if the operations performed result to 24.

        States 
            {"r": 0} : Not finished.
            {"r": 1} : Finished and correct.
            {"r": -1} : Finished and incorrect.
        """
        if any(["_" in ans for ans in state.ans]):
            return {"r":0}
        elif state.board == state.board_gt:
            return {"r":1}
        else:
            return {"r":0}
    
    @staticmethod
    def get_metrics(experiment_logs):
        """
        Given a file path, return the metrics of the experiment based on the logs.
        """
        #with open(file_path, "r") as f:
        #    data = json.load(f)

        experiment_logs = experiment_logs.copy() # Making sure initial logs are not modified
        info = experiment_logs.pop("Info")

        metrics = {}
        averages = []
        for puzzle_id, puzzle_results in experiment_logs.items():
            initial_puzzle = puzzle_results.pop("puzzle")       # Not needed just want to pop
            verifications = puzzle_results.pop("Verifications") # Not needed just want to pop

            max_actions = 0
            actions2metrics = {}
            for agent_id, agent_results in puzzle_results.items():
                for step_id, step_results in agent_results.items():
                    step_actions = len(step_results["Step"].split(" -> "))
                    actions2metrics.setdefault(step_actions, []).append(step_results["metrics"])
                    if step_actions > max_actions:
                        max_actions = step_actions
                        metrics[puzzle_id] = step_results["metrics"]
            assert max_actions > 0, f"No actions found for {puzzle_id}"

            # Averaging the metrics across agents
            max_num_actions = max(actions2metrics.keys())
            data = actions2metrics[max_actions]
            averages_puzzle = {key: sum(d[key] for d in data) / len(data) for key in data[0]}
            averages.append(averages_puzzle)

        
        r_letter = mean([metric["r_letter"] for metric in metrics.values()])
        r_word = mean([metric["r_word"] for metric in metrics.values()])
        r_all = mean([metric["r_all"] for metric in metrics.values()])

        results = {"r_letter": r_letter, "r_word": r_word, "r_all": r_all}

        overall = {key + "_av": sum(d[key] for d in averages) / len(averages) for key in averages[0]}

        results.update(overall)
        

        return results
        
        

def parse_line(input_str):
    # regular expression pattern to match the input string format
    pattern = r'^([hv][1-5])\. ([a-zA-Z]{5,5}) \((certain|high|medium|low)\).*$'

    # use regex to extract the parts of the input string
    match = re.match(pattern, input_str)

    if match:
        # extract the matched groups
        parts = [match.group(1), match.group(2), match.group(3)]
        return parts
    else:
        return None

def parse_response(response):

    # map confidence levels to values
    confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}  # TODO: ad hoc

    # split the response into lines
    lines = response.split('\n')

    # parse each line
    parsed_lines = [parse_line(line) for line in lines]

    # filter out the lines that didn't match the format
    parsed_lines = [(line[0].lower() + '. ' + line[1].lower(), confidence_to_value.get(line[2], 0)) for line in parsed_lines if line is not None]

    return parsed_lines if len(parsed_lines) >= 1 else None

def provokes_change(state, action):
    """
    Given  a state and an action return whether the action provokes a change to the state's board.
    """
    current_board = state.board.copy()
    new_board = state.board.copy()

    action = parse_action(action)
    pos, word = action

    # Update new board based on the action
    if pos.startswith('h'):
        idx = int(pos[1:]) - 1
        new_board[idx*5:(idx+1)*5] = list(word.upper())
    elif pos.startswith('v'):
        idx = int(pos[1:]) - 1
        new_board[idx::5] = list(word.upper())
        idx += 5  # for later status update
    else:
        return False
    if new_board == current_board:
        return False
    else:
        return True
    
def parse_action(action: str)-> str:
    action = action.split('\n')[-1]
    action = action.split('. ')
    return action
