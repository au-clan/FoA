# Updated
bfs_prompt = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list possible next steps as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)

Input: {input}
Possible next steps:
'''

bfs_reflexion_prompt = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list possible next steps as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)



Based on previous attempts to solve the puzzle, here is some advice on how to proceed:
{reflexion}

Input: {input}
Possible next steps:'''

reflexion_step_prompt = '''The game of 24 is a math puzzle where players use four numbers and basic arithmetic operations (+ - * /) to make the result equal to 24. Following is a single step, which was determined to have failed
Input: {puzzle}
Step attempt:
{steps}

Reflect on the previous attempt and provide a reflection below:
- If there's a mistake, identify it and explain how similar mistakes can be avoided.
- If the mistake can be generalized, provide a general reflection.
- Be succint and clear in your reflection.
- Do not provide a new solution, only a reflection.

Reflection:
'''

reflexion_prompt = '''The game of 24 is a math puzzle where players use four numbers and basic arithmetic operations (+ - * /) to make the result equal to 24. Following is a previous attempt at solving the puzzle.
Input: {puzzle}
Solution attempt:
{steps}


Reflect on the previous attempt and provide a reflection below:
- If there's a mistake, identify it and explain how similar mistakes can be avoided.
- If the mistake can be generalized, provide a general reflection.
- Be succint and clear in your reflection.
- Do not provide a new solution, only a reflection.

Reflection:
'''

summary_prompt = '''The game of 24 is a math puzzle where players use four numbers and basic arithmetic operations (+ - * /) to make the result equal to 24. Following is a previous attempt at solving the puzzle.

You made the following list of reflections:

{reflexion}

Summarize all the reflexions and discard duplicates

Summarization of all reflections:
'''

#Rafa prompt
propose_prompt = '''Now use numbers and basic arithmetic operations (+ - * /) to generate possible next steps. Make sure use steps that is sure to leads to 24 and avoid steps that are impossible to generate 24. Note that it is possible that we are considering intermediate steps so the numbers of the input may be less than 4.
Example:
Input: 2 8 8 14 
Possible next steps: 
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Example:
Input: 2 5 8
5 - 2 = 3 (left: 3 8)
5 * 2 = 10 (left: 10 8)
8 / 2 = 4 (left: 4 5)
Now try with the following input:
Input: {input}
Possible next steps:
'''

#Prompt with only one suggestion for testing
bfs_prompt_single = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list a possible next step as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)

Input: {input}
A possible next step:
'''

#Rafa prompt
propose_prompt = '''Now use numbers and basic arithmetic operations (+ - * /) to generate possible next steps. Make sure use steps that is sure to leads to 24 and avoid steps that are impossible to generate 24. Note that it is possible that we are considering intermediate steps so the numbers of the input may be less than 4.
Example:
Input: 2 8 8 14 
Possible next steps: 
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Example:
Input: 2 5 8
5 - 2 = 3 (left: 3 8)
5 * 2 = 10 (left: 10 8)
8 / 2 = 4 (left: 4 5)
Now try with the following input:
Input: {input}
Possible next steps:
'''

#Prompt with only one suggestion for testing
bfs_prompt_single = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list a possible next step as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)

Input: {input}
A possible next step:
'''

bfs_reflexion_prompt_single = '''Use numbers and basic arithmetic operations (+ - * /). Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list, a possible next step as well as all the remaining numbers and nothing else.

Example: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)

Example: 1 3
Possible next steps:
1 + 3 = 4 (left: 4)
1 * 3 = 3 (left: 3)
3 - 1 = 2 (left: 2)
3 / 1 = 3 (left: 3)
1 - 3 = -2 (left: -2)



Based on previous attempts to solve the puzzle, here is some advice on how to proceed:
{reflexion}

Input: {input}
A possible next step:'''

evaluate_prompt = '''The game of 24 is a math puzzle where players use four numbers and basic arithmetic operations (+ - * /) to make the result equal to 24. Following is a previous attempt at solving the puzzle.
Input: {puzzle}
Solution attempt:
{steps}


Task:
1. Evaluate step
   - Evaluate if the given numbers at each step can reach 24 for each step with the following sure, likely or impossible

   Examples for evaluating:
    10 14
    10 + 14 = 24
    sure
    11 12
    11 + 12 = 23
    12 - 11 = 1
    11 * 12 = 132
    11 / 12 = 0.91
    impossible
    4 4 10
    4 + 4 + 10 = 8 + 10 = 18
    4 * 10 - 4 = 40 - 4 = 36
    (10 - 4) * 4 = 6 * 4 = 24
    sure
    4 9 11
    9 + 11 + 4 = 20 + 4 = 24
    sure
    5 7 8
    5 + 7 + 8 = 12 + 8 = 20
    (8 - 5) * 7 = 3 * 7 = 21
    I cannot obtain 24 now, but numbers are within a reasonable range
    likely
    5 6 6
    5 + 6 + 6 = 17
    (6 - 5) * 6 = 1 * 6 = 6
    I cannot obtain 24 now, but numbers are within a reasonable range
    likely
    10 10 11
    10 + 10 + 11 = 31
    (11 - 10) * 10 = 10
    10 10 10 are all too big
    impossible
    1 3 3
    1 * 3 * 3 = 9
    (1 + 3) * 3 = 12
    1 3 3 are all too small
    impossible
2. Check whether each step is valid.
   - Verify if the arithmetic is correct (e.g., 4 * 6 = 24, 8 - 3 = 5, etc.).
   - Verify if the step uses numbers that are still available.
   - Verify if the result of each step is computed correctly and is used in subsequent steps properly.

IMPORTANT: I want you to end your response with stating what step went wrong (0 indexed) for example: "Incorrect step: 2"
'''

#RAFA prompt
validation_prompt = '''Evaluate if given formula is a valid move in the game of 24. Especially, check if a number is missing, if the arithmetic is incorrect, or if a number is used that is not in the input or used twice. All four numbers does not need to be used for the first three steps. Always end your answer with Invalid or Valid.
Example

Input: 3 6 8 10
3 * 6 = 18 (left: 18 8 10)
Valid

Input: 2 6 8 14
2 * 6 = 1 (left: 1 8 14)
Invalid

Input: 4 6 8 10
10 * 5 = 50 (left: 6 50)
Invalid

Input: 1 5 7
5 * 5 = 25 (left: 1 25 7)
Invalid

Now evaluate the followng formula:
Input: {puzzle}
{steps}
'''
# Updated
value_prompt = '''Evaluate if given numbers can reach 24 by responding with the following Sure, Likely or Impossible.

Examples:
10 14
10 + 14 = 24
Sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
Impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
Sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
Sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
Likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
Likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
Impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
Impossible

Input: {steps}
'''



# Taken from Tree of Thoughts paper
cot_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Return only the complete answer.

Example: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Example: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Example: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Example: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Example: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24

Input: {input}
'''

# Taken from Tree of Thoughts paper
value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24. Do not explain simply list the judgement.
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure
Example: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure
Example: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure
Example: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible
Example: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible
Example: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible
Input: {input}
Answer: {answer}
Judge:'''