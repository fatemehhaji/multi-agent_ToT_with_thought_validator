import os
import re
import json
from src.tot.tasks.base import Task, DATA_PATH
from src.tot.prompts.gsm8k import *
from src.tot.models import gpt
import random

def read_jsonl(path: str):
    with open(path) as fh:
        random_seed = 10
        random.seed(random_seed)
        data = [json.loads(line) for line in fh.readlines() if line]
        random.shuffle(data)
        return data
    

class GSM8KTask(Task):
    """
    Input (x)   : a math word problem
    Output (y)  : a numerical answer
    Reward (r)  : 1 if the answer is correct, 0 otherwise
    """
    def __init__(self, file='test.jsonl'):
        """
        file: a JSON Lines file, each line containing a math problem
        """
        super().__init__()
        path = os.path.join(DATA_PATH, 'gsm8k', file)
        self.data = read_jsonl(path)
        self.steps = 2
        self.stops = ['\nPassage:\n', None]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]['question']
    
    @staticmethod
    def filter_output(output):
        # Find the indices of the occurrences of "Strategy:"
        indices = [m.start() for m in re.finditer(r"Strategy:", output)]
        if len(indices) < 2:
            # If less than two "Strategy:" found, use the whole output
            output_to_search = output
        else:
            # Use the output starting from the second "Strategy:"
            second_strategy = indices[1]
            output_to_search = output[second_strategy:]

        # Define the pattern to search for a numeric answer after "the answer is"
        pattern = r"the answer is[^\d]*(\d+(?:,\d+)*(?:\.\d+)?)"
        match = re.search(pattern, output_to_search, re.IGNORECASE)

        if match:
            # Extract and return the numeric part of the answer, replacing commas
            numeric_answer = match.group(1).replace(',', '')
            return numeric_answer
        
        # If no match found in the second part, search in the first part if it exists
        if len(indices) > 0 and indices[0] > 0:  # Ensure there is a first part to search
            first_part = output[:indices[0]]  # Search before the first occurrence
            match = re.search(pattern, first_part, re.IGNORECASE)
            if match:
                # Extract and return the numeric part of the answer, replacing commas
                numeric_answer = match.group(1).replace(',', '')
                return numeric_answer

        # If no match is found, return None
        return None

    def test_output(self, idx: int, output: str):
        correct_answer = re.search(r'#### (\d+(?:\.\d+)?)', self.data[idx]['answer'])
        if correct_answer:
            correct_answer = correct_answer.group(1)
            filtered_output = self.filter_output(output)
            if filtered_output is not None:
                # Compare as floats to handle decimal answers
                is_correct = abs(float(filtered_output) - float(correct_answer)) < 1e-6
                info = {"r": 1 if is_correct else 0}
            else:
                info = {"r": 0}  # If we can't extract an answer, assume it's wrong
        else:
            info = {"r": 0}  # If we can't find the correct answer, assume it's wrong
        return info


    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1