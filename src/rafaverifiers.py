import itertools
import re
import sympy
from typing import Tuple

class RafaVerifier():
    def check_answer(self, problem: str, answer: str):
        expression = answer.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', problem)
        if sorted(numbers) != sorted(problem_numbers):
            return False, "The numbers you use are not the original numbers from the problem."
        try:
            # print(sympy.simplify(expression))
            if sympy.simplify(expression) == 24:
                return True, "The formula is correct."
            else:
                return False, "The formula does not lead to 24."

        except Exception as e:
            # print(e)
            return False, "The formula is invalid."


    def check_valid_move(self, idx, last_step, cur_step):
        if idx == 1:
            original_nums = [float(num) for num in last_step.split(" ")]
        else:
            original_nums = [float(num) for num in last_step.split('left:')[-1].strip("()").split(" ") if
                            num != '']
        formula = [op for op in cur_step.split('left:')[0].strip("()").split(" ") if op != '']
        new_nums = [float(num) for num in cur_step.split('left:')[-1].strip().strip("()").split(" ") if num != ''] #Added a strip() to remove potential whitespace behind "(left: x)  "
        

        try:
            print(original_nums, new_nums, formula)
            original_nums.remove(float(eval(formula[0])))
            original_nums.remove(float(eval(formula[2])))
            for num in original_nums:
                new_nums.remove(num)
            new_nums.remove(float(formula[4]))
            assert len(new_nums) == 0
        except ValueError:
            return False, "You use value that does not exists in last step or you use them repeatedly; or you drop numbers from the last step."
        except AssertionError:
            return False, "You have more numbers left than expected."

        return True, "The move the valid and correct."


    def check_equation(self, equation):
        try:
            left, right = equation.split("=")
            err = abs(eval(left) - float(right))
            if err < 1e-10:
                return True, "The Equation is correct."
            else:
                return False, f"The Equation is incorrect, the result should be {eval(left)}"
        except Exception as e:
            print(e)
            return False, "The Equation is not valid."


    def check_twentyfour(self, cur_step):
        cards = [float(num) for num in cur_step.split('left:')[-1].strip("()").split(" ") if num != '']

        try:
            for nums in itertools.permutations(cards):  # 四个数
                for ops in itertools.product('+-*/', repeat=len(cards) - 1):  # 三个运算符（可重复！）
                    # 构造三种中缀表达式 (bsd)
                    if len(cards) == 4:
                        bds1 = '({0}{4}{1}){5}({2}{6}{3})'.format(*nums, *ops)  # (a+b)*(c-d)
                        bds2 = '(({0}{4}{1}){5}{2}){6}{3}'.format(*nums, *ops)  # (a+b)*c-d
                        bds3 = '{0}{4}({1}{5}({2}{6}{3}))'.format(*nums, *ops)  # a/(b-(c/d))
                        bdss = [bds1, bds2, bds3]
                    elif len(cards) == 3:
                        bds1 = '({0}{3}{1}){4}{2}'.format(*nums, *ops)  # (a+b)*c
                        bds2 = '{0}{3}({1}{4}{2})'.format(*nums, *ops)  # a+(b*c)
                        bdss = [bds1, bds2]
                    elif len(cards) == 2:
                        bds1 = '({0}{2}{1})'.format(*nums, *ops)  # a+b
                        bdss = [bds1]
                    else:
                        if len(nums) == 1 and abs(nums[0] - 24) < 1e-5:
                            return True, ""
                        return False, ""
                    for bds in bdss:  # 遍历
                        try:
                            if abs(eval(bds) - 24.0) < 1e-10:  # eval函数
                                return True, ""
                        except ZeroDivisionError:  # 零除错误！
                            continue
        except Exception as e:
            print(e)
            return False, "It is not a valid formula."
        return False, ""

    #Modified RAFA's check_step, to match how we have designed states in FOA
    def check_all(self, state, last_step) -> Tuple[bool, str]:
        idx = len(state.steps)
        cur_step = state.steps[-1]
        print("idx: ", idx)
        print("cur_step: ", cur_step)
        print("last_step: ", last_step)
        print((idx, last_step, cur_step))
        try:
            if "answer" in cur_step.lower():
                #print("Current step contains answer thus Checking answer")
                correct, feedback = self.check_answer(state.puzzle, cur_step)
                #print("Answer is correct: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx-1} tries to give an answer but it is incorrect. {feedback}", 0
                return f"Step {idx-1} is correct. {feedback}", 10
            else:
                # Check if the step is valid
                #print("Current step does not contain answer, Checking valid move")
                correct, feedback = self.check_valid_move(idx, last_step, cur_step)
                #print("Current step is valid: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx-1} is illegal. {feedback}", 0

                formula = cur_step.split('left:')[0].strip("()")
                #print("Checking equation, formula:", formula)
                correct, feedback = self.check_equation(formula)
                #print("Equation is correct: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx-1} is not correctly calculated. {feedback}", 0
                #print("Checking if it is possible to reach 24")
                correct, feedback = self.check_twentyfour(cur_step)
                #print("It is possible: ", correct, " Why: ", feedback)
                if not correct:
                    return f"Step {idx-1} is impossible to lead to 24. {feedback}", 0

                return f"Step {idx-1} is correct and can lead to 24.", 1
            
        except Exception as e:
            print(e)
            return f"Step {idx-1} is invalid.", 0
