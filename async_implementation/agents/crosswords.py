import re
import asyncio
import random
from typing import Tuple

from async_implementation.prompts import crosswords as prompts
from async_implementation.states.crosswords import CrosswordsState

class CrosswordsAgent:

    @staticmethod
    async def get_candidates(state: CrosswordsState, api, n:int =8)-> dict:
        """
        Given a state, return a dictionary of candidate actions along with its scores.
        """
        
        # Render the state
        obs = CrosswordsState.render(state)

        # Get candidate actions and their scores
        prompt = prompts.propose_prompt.format(input=obs)
        coroutines = []
        for _ in range(n):
            coroutines.append(api.buffered_request(prompt))
        responses = await asyncio.gather(*coroutines)

        # Parse the responses and add the scores for each action
        candidates_to_scores = {}
        for response in responses:
            parsed_response = parse_response(response)
            if parsed_response:
                for candidate, score in parsed_response:
                    candidates_to_scores[candidate] = candidates_to_scores.get(candidate, 0) + score
        return candidates_to_scores
    
    @staticmethod
    async def step(state: CrosswordsState, api)-> Tuple[CrosswordsState, str]:
        """
        Given a state, return the next state.
        """
        
        # Get next step suggestions/actions and pick one of the ones with the highest value
        suggestions = await CrosswordsAgent.get_candidates(state, api)
        assert len(suggestions) > 0, "No suggestions found" # -> TODO: Can deal with this by resampling the specific agent
        suggestions_max_value = max(suggestions.values())
        max_value_suggestions = [suggestion for suggestion, value in suggestions.items() if value == suggestions_max_value]
        random.seed(state.randomness)
        action = random.choice(max_value_suggestions)

        #Parse the action
        action = action.split('\n')[-1]
        action = action.split('. ')

        # Get the position and word from the action
        pos, word = action
        
        # Assert the action is valid TODO: Maybe change to actual assert (?)
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}
        
        # New board = Current board, before the action is implemented
        new_board = state.board.copy()

        # Update new board based on the action
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            new_board[idx*5:(idx+1)*5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            new_board[idx::5] = list(word.upper())
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        # Get new answer and status based on the current board
        new_ans = CrosswordsState.get_ans(new_board)
        new_status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(state.status, state.ans, new_ans)]
        new_status[idx] = 1

        # Return the next state
        random.seed(state.randomness)
        next_state = CrosswordsState(
            data=state.data,
            board_gt=state.board_gt,
            ans_gt=state.ans_gt,
            board=new_board, 
            ans=new_ans, 
            status=new_status,
            steps = state.steps + [" .".join(action)],
            randomness=random.randint(0, 1000)
             )
        return next_state, action
        
    @staticmethod
    async def evaluate(state, api):
        """
        Evaluates the current state and returns a value number.
        The state is evaluated line by line.
        """

        # Count of the results for each line
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        
        for ans, data in zip(state.ans, state.data):
            if ans.count('_') >= 4:continue
            
            # Parse the answer along with the original question
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            
            # Get a value for the line from the set {sure, maybe, impossible}
            prompt = prompts.value_prompt.format(input=line)
            res = await api.buffered_request(prompt)
            
            # Parse the response and update the count
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1

        # Map the count to a value number    
        value_map = {'impossible': -20, 'maybe': 5, 'sure': 20} #TODO: ad hoc
        value_number  =sum(value * value_map[name] for name, value in count.items())
        return value_number
        

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