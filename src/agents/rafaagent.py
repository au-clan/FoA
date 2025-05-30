from . import *
import itertools
import numpy as np
from src.prompts.adapt import gameof24 as prompts
from functools import partial
import re

def get_value(env, history, x, y, n_evaluate_sample, cache_value=True):
    # validation_prompt = env.validation_prompt_wrap(x, y)
    # if validation_prompt:
    #     validation_outputs = gpt_with_history(validation_prompt, history, n=1, stop=None)
    #     validation = env.validation_outputs_unwrap(x, y, validation_outputs)
    #     if validation == 0:
    #         return 0
    value_prompt = env.value_prompt_wrap(x, y)
    if cache_value and value_prompt in env.value_cache:
        #print("value was in cache")
        return env.value_cache[value_prompt]
    print("Calls gpt with history")
    value_outputs = gpt_with_history(value_prompt, history, temperature=0.3, n=n_evaluate_sample, stop=None)
    print("value_outputs: ", value_outputs)
    value = env.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        env.value_cache[value_prompt] = value
    print("value: ", value)
    return value

def get_values(env, history, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache and cache_value:  # avoid duplicate candidates
            value = local_value_cache[y]
            #print("Y was in cache")
        else:
            #print("Y was not in cache, calls get_value()")
            value = get_value(env, history, x, y, n_evaluate_sample, cache_value=cache_value)
            if cache_value:
                local_value_cache[y] = value
        values.append(value)
    return values

def get_proposals(env, history, x, y, n_propose_sample=10):
    propose_prompt = env.propose_prompt_wrap(x, y)
    #print("x:", x)
    proposal_list = [x.split('\n') for x in gpt_with_history(propose_prompt, history, n=1, stop=["\n\n"])]
    #for xx in gpt_with_history(propose_prompt, history, n=1, stop=["\n\n"]):
    #    print("x in for: ", xx)
    proposals = []
    for p in proposal_list:
        proposals.extend(p)
    #print("proposals before indexing: ", proposals)
    proposals = proposals[:min(len(proposals), n_propose_sample)]
    proposals = [p.replace('\u00f7', '/') for p in proposals]
    print("proposals: ", proposals)
    return [y + _ + '\n' for _ in proposals]

def get_proposal(env, history, x, y):
    propose_prompt = env.single_proposal_prompt_wrap(x,y)
    print("propose_prompt ", propose_prompt)
    proposal_list = [x.split('\n') for x in gpt_with_history(propose_prompt, history, n=1, stop=["\n\n"])]
    proposals = []
    for p in proposal_list:
        proposals.extend(p)
    print("proposals: ", proposals)
    proposals = [p.replace('\u00f7', '/') for p in proposals]
    return [y + _ + '\n' for _ in proposals]

class TreeOfThoughtAgent(Agent):
    def __init__(self, backend, temperature, prompt_sample, method_generate, method_evaluate, method_select, method_reflexion_type,
                 n_generate_sample,
                 n_evaluate_sample, n_select_sample, k, limit):
        super().__init__()

        global gpt
        global gpt_with_history
        gpt = partial(gpt, model=backend)
        gpt_with_history = partial(gpt_with_history, model=backend)

        self.backend = backend
        self.prompt_sample = prompt_sample
        self.method_generate = method_generate
        self.method_evaluate = method_evaluate
        self.method_select = method_select
        self.method_reflexion_type = method_reflexion_type #Added new method reflexion type
        self.n_generate_sample = n_generate_sample
        self.n_evaluate_sample = n_evaluate_sample
        self.n_select_sample = n_select_sample
        self.k = k 
        self.limit = limit
        self.reflects = []
        self.all_reflects = []
        self.value_reflects = []

    def plan(self, env, to_print=True):
        print(gpt)
        print(gpt_with_history)
        x = env.puzzle  # input
        print("current puzzle is ", x)
        history = env.history  # history
        print("history: ", history)
        ys = ["\n".join(history) + "\n"] if len(history) else [""]  # current output candidates
        print("ys: ", ys)
        infos = []
        prompt = "Now we would like to play a game of 24. That is, given 4 numbers, try to use "
        "them with arithmetic operations (+ - * /) to get 24. "
        obs = [{"feedback":prompt},
               {"feedback": "What you have learned about the puzzle are summarized below.\n" + "\n".join(
                   self.reflects)}]
        value_obs = [prompt,
                     dict(feedback="What you have learned about the puzzle are summarized below.\n" + "\n".join(
                         self.value_reflects))]
        print("self.value_reflects", value_obs[1])
        for step in range(4 - len(history)):
            print("step: ", step)
            # generation
            new_ys = [get_proposals(env, obs, x, y, self.n_generate_sample) for y in ys]
            #elif self.method_generate == "single":
                #new_ys = [get_proposal(env, obs, x, y) for y in ys]
            new_ys = list(itertools.chain(*new_ys))
            #print("new_ys: ", new_ys)
            ids = list(range(len(new_ys)))
            #print("ids: ", ids)
            # evaluation
            if self.method_generate == "propose":
                values = get_values(env, value_obs, x, new_ys, self.n_evaluate_sample, cache_value=False)  
                # selection
                select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:self.n_select_sample]
                select_new_ys = [new_ys[select_id] for select_id in select_ids]
            elif self.method_generate == "single":
                select_new_ys = [new_ys[0]] if new_ys else []
                values = [0] * len(new_ys)
            print("selected new ys: ", select_new_ys) 
            # log
            if to_print:
                sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
                print(
                    f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

            infos.append(
                {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
            ys = select_new_ys
        print('ys before ys_list: ', ys)
        res_ys = "\n".join(y.strip() for y in ys[0].splitlines()) # Splitting the ys list at every \n, then stripping away trailing and leading whitespace
        print("res_ys: ", repr(res_ys))
        return res_ys, {'steps': infos}

    def reflect(self, env, obs):
        y = obs['answer']
        feedback = obs['feedback']
        reflect_prompt, value_reflect_prompt = env.reflect_prompt_wrap(env.puzzle, y, feedback)
        reflects = gpt(reflect_prompt, stop=None)
        value_reflects = gpt(value_reflect_prompt, stop=None)
        if self.method_reflexion_type == "list":
            self.reflects.extend(reflects)
            print("self.reflects: ", self.reflects)
            self.value_reflects.extend(value_reflects)
            print("self.value_reflects: ", self.value_reflects)
        if self.method_reflexion_type == "k_most_recent":
            self.reflects.extend(reflects)
            self.value_reflects.extend(value_reflects)
            print("self.reflects before k: ", self.reflects)
            print("self.value_reflects before k: ", self.value_reflects)
            if len(self.reflects) > self.k:
                self.reflects.pop(0)
            if len(self.value_reflects) > self.k:
                self.value_reflects.pop(0)
            print("self.reflects after k: ", self.reflects)
            print("self.value_reflects after k: ", self.value_reflects)
        if self.method_reflexion_type == "summary":
            # Step 1: Extend and summarize
            self.all_reflects.extend(reflects)
            self.value_reflects.extend(value_reflects)
            summary_prompt = env.summary_prompt_wrap(self.all_reflects, self.limit)
            summary = gpt(summary_prompt, stop=None)
            self.reflects = summary
            print("all reflexions: ", self.all_reflects)
            print("summary: ", self.reflects)
            print("self.value_reflects: ", self.value_reflects)

            # Step 2: Normalize and clean
            summarized_labels = [
                entry.strip() for entry in self.value_reflects
                if re.match(r'^\d+(?:\s+\d+)*:\s*(sure|impossible)$', entry.strip())
            ]
            text_reflections = [entry for entry in self.value_reflects if entry not in summarized_labels]

            # Step 3: Normalize commas to avoid broken matches (e.g., '1, 5, 6: sure')
            text = " ".join(text_reflections)
            text = re.sub(r'(?<=\d),\s*(?=\d)', ' ', text)

            # Step 4: Define patterns
            patterns = [
                r'\(left:\s*([\d\s]+)\)\s*[:\-–]?\s*(sure|impossible)',  # Old format
                r'\b(\d+(?:\s+\d+)*):\s*(sure|impossible)'               # New format
            ]

            # Step 5: Extract and deduplicate
            new_labels = []
            for pattern in patterns:
                matches = re.findall(pattern, text)
                for nums, label in matches:
                    formatted = f"{nums.strip()}: {label}"
                    if formatted not in summarized_labels and formatted not in new_labels:
                        new_labels.append(formatted)

            # Step 6: Combine all labels
            self.value_reflects = summarized_labels + new_labels

            print("self.value_reflects after summary: ", self.value_reflects)

        return

    def act(self, env, obs):
        print(obs['feedback'])
        if len(obs['feedback']) >= 1:
            self.reflect(env, obs)
        action, info = self.plan(env)
        return action,info

    def update(self, obs, reward, done, info):
        if done:
            self.reflects = []
            self.all_reflects = []
            self.value_reflects = []