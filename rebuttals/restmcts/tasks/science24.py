import re
import os
from tasks.prompts import *


# data: question: str
# mode: 'cot', 'tot', 'mcts'
# method: 'glm', 'gpt', 'local'
class SearchTask24(object):
    def __init__(self, data, propose_method='glm', value_method='glm'):
        super().__init__()
        self.question = data
        self.propose_method = propose_method
        self.value_method = value_method
        self.value_cache = {}

    def clear_cache(self):
        self.value_cache = {}

    @staticmethod
    def single_reflection_wrap_simple(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'reflection', '==============================', '\nstep: ', step)
        #print('propose_prompt: \n', x + '\nExisting steps:\n' + y + 'Opinions based on the above steps:\n')
        if lang == 'zh':
            assert False
            # if not y:
            #     y = '无\n'
            # prompt = single_reflection_prompt_simple + x + '\n已有步骤:\n' + y + '\n输出:'  # simple style
        else:
            if not y:
                y = 'None\n'
            prompt = single_reflection_prompt_simple_en + f"Use numbers and basic arithmetic operations (+ - * /) to obtain 24.\n"
            prompt += "Input : " + x + '\nExisting Steps:\n' + y.strip() + '\nOutput:'
        return prompt
    
    @staticmethod
    def zero_single_propose_wrap_gpt(x: str, y: str = '', step: int = 0, lang: str = 'zh') -> str:
        print('\n', '==============================', 'proposal', '==============================', '\nstep: ', step)
        #print('propose_prompt: \n', x + '\nExisting steps:\n' + y + 'Based on the above steps, the possible solution for the current step is:\n')
        if lang == 'zh':
            assert False
            # if not y:
            #     y = '无\n'
            # prompt = zero_single_proposal_prompt_gpt + x + '\n已有步骤:\n' + y + '\n输出:'
        else:
            if y=='':
                prompt = zero_single_proposal_prompt_gpt_en24.format(input=x)
            else:
                print('y:', y)
                input = y.strip().split("\n")[-1].split("left: ")[-1][:-1]
                if input == "24":
                    prompt = zero_single_proposal_prompt_gpt_en24cot.format(input=x) + 'Steps\n' + y.strip()
                else:
                    prompt = zero_single_proposal_prompt_gpt_en24.format(input=input)
        return prompt
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        print('\n', '==============================', 'critic', '==============================', '\n')
        last_step = y.strip().split('\n')[-1]
        if 'left' not in last_step.lower():
            ans = last_step.lower().replace('answer: ', '')
            value_prompt = critic_simplified24final.format(input=x, answer=ans)
        else:
            current_numbers = last_step.split("left: ")[-1][:-1]
            value_prompt = critic_simplified24.format(input=current_numbers)
        return value_prompt
    
    @staticmethod
    def MATH_summary_prompt_wrap(x: str, y: str = '') -> str:
        print('\n', '==============================', 'summary', '==============================', '\n')
        #print('summary_prompt: \n', x + '\n已有步骤:\n' + y + '基于以上步骤的综述为:\n')
        prompt = MATH_summary_prompt + f"Use numbers and basic arithmetic operations (+ - * /) to obtain 24.\n" + "Input : " + x + '\nSolution: ' + y.strip() + '\nExtracted answer:'
        return prompt

    @staticmethod
    def value_outputs_unwrap(value_outputs: list, low=0.0, high=1.0) -> float:
        out_value = low
        all_out = ''
        for _ in value_outputs:
            all_out = all_out + _
        # '分数' = "Fraction" -> Expected output ends with eg. "分数: 0.3"
        if '分数' not in all_out:
            print('分数输出不合法!\n')
            return out_value
        stp = all_out.split('分数')[-1].strip()
        try:
            match = re.findall(r'-?[0-9]+\.?[0-9]*', stp)[-1]
            out_value = float(match)
            out_value = min(max(low, out_value), high)
        except Exception as e:
            print(f'The score output is wrong! Error type:{e}\n')
            return low
        return out_value
    
    @staticmethod
    def value_outputs_unwrap(value_outputs: list, y: str, low=0.0, high=1.0)-> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return low
        try:
            value_names = [_.split('\n')[-1] for _ in value_outputs]
            value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
            value = sum(value * value_names.count(name) for name, value in value_map.items())
        except Exception as e:
            print(f'The score output is wrong! Error type:{e}\n')
            value=low
        return value