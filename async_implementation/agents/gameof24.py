import random

from async_implementation.prompts import gameof24 as prompts
from async_implementation.states.gameof24 import GameOf24State


class GameOf24Agent:

    @staticmethod
    async def step(state: GameOf24State, api, limiter):
        """
        Given a state, return the next state.
        """

        # set up the prompt, based on the current state
        current_state = state.current_state
        prompt = prompts.bfs_prompt.format(input=current_state)

        # Get the next state
        messages = [{"role":"user", "content":prompt}]
        iid_suggestions = await api.request(messages, limiter, n=1)
        suggestions = iid_suggestions[0]

        # parse suggestions
        suggestions = suggestions.split("\n")
        selected_suggestion = random.choice(suggestions)
        selected_state = GameOf24Agent.parse_next_state(selected_suggestion)

        # set up new state object
        next_state = GameOf24State(
            current_state=selected_state,
            steps=state.steps + [selected_suggestion]
        )
        return next_state

    @staticmethod
    def parse_next_state(suggestion: str) -> str:
        return suggestion.split('left: ')[-1].split(')')[0]
