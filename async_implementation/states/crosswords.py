import json
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class CrosswordsState:

    data: List[str] 
    board_gt: List[str] 
    ans_gt: List[str]
    randomness: int

    # State with no steps taken / initial state
    board: List[str] = field(default_factory=lambda: ["_"]*25)
    ans : List[str] = field(default_factory=lambda: ["_____"]*10)
    status: List[int] = field(default_factory=lambda: [0]*10)
    
    
    @staticmethod
    def render_board(board: List[str])-> List[str]:
        """
        Renders the board.

        Example
            boardt: ['A', 'G', 'E', 'N', 'D', 'M', 'O', 'T', 'O', 'R', 'A', 'R', 'T', 'S', 'Y', 'S', 'A', 'L', 'L', 'E', 'S', 'L', 'E', 'E', 'R']
            return: GT Board:
                    A G E N D
                    M O T O R
                    A R T S Y
                    S A L L E
                    S L E E R
        """
        s=""
        for i in range(5):
            s += ' '.join(board[i*5:(i+1)*5]) + '\n'
        return s


    @staticmethod
    def render_clues(data, status=None, state_status=None):
        """
        Renders the data/clues. If status is not None, only render the clues/data with state_status equal to status.

        Example (first 3)
            data: ['An agendum; something to be done', 'An engine', 'Pretentious; flowery']
            return: h1. An agendum; something to be done
                    h2. An engine
                    h3. Pretentious; flowery
        """
        s = ""
        for i in range(5):
            if status is None or state_status[i] == status:
                s += 'h' + str(i+1) + '. ' + data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or state_status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + data[i] + '\n'
        return s
    

    @staticmethod
    def render_ans(data, ans, status=None, state_status=None):
            """
            Renders the answers. If status is not None, only render the answers with state_status equal to status.

            Example (first 3)
                data: ['An agendum; something to be done', 'An engine', 'Pretentious; flowery']
                ans: ['AGEND','MOTOR','ARTSY']
                return: h1. An agendum; something to be done: AGEND
                        h2. An engine: MOTOR
                        h3. Pretentious; flowery: ARTSY
            """
            s = ""
            # s += "Horizontal:\n"
            for i in range(5):
                if status is None or state_status[i] == status:
                    s += 'h' + str(i+1) + '. ' + data[i] + ': ' + ans[i] + '\n'
            # s += "Vertical:\n"
            for i in range(5, 10):
                if status is None or state_status[i] == status:
                    s += 'v' + str(i-5+1) + '. ' + data[i] + ': ' + ans[i] + '\n'
            return s 
    

    @staticmethod
    def get_ans(board: List[str])-> List[str]:
        """"
        Given the board, return the answers.

        Example
            board: ['A', 'G', 'E', 'N', 'D', 'M', 'O', 'T', 'O', 'R', 'A', 'R', 'T', 'S', 'Y', 'S', 'A', 'L', 'L', 'E', 'S', 'L', 'E', 'E', 'R']
            ans: ['AGEND', 'MOTOR', 'ARTSY', 'SALLE', 'SLEER', 'AMASS', 'GORAL', 'ETTLE', 'NOSLE', 'DRYER']
        """
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    
    @staticmethod
    def render(state, status=True):
        ans = CrosswordsState.get_ans(state.board)
        s = ""
        s += CrosswordsState.render_board(state.board)
        if status:
            # Returns answers discriminating them based on status
            s += "\nUnfilled:\n"
            s += CrosswordsState.render_ans(state.data, ans, status=0, state_status=state.status)
            s += "\nFilled:\n"
            s += CrosswordsState.render_ans(state.data, ans, status=1, state_status=state.status)
            s += "\nChanged:\n"
            s += CrosswordsState.render_ans(state.data, ans, status=2, state_status=state.status)
        else:
            # Returns answers without status discrimination
            s+= "\n"
            s+= CrosswordsState.render_ans(state.data, ans)
        return s

