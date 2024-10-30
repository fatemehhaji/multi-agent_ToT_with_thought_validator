import re
from tasks.base import Task, DATA_PATH
from prompts.gsm8k import *
from models.models import gpt
    
class GSM8KTask(Task):
    """
    Input (x)   : a problem
    Output (y)  : "true" or "false"
    Reward (r)  : # TODO (if applicable)
    """
    def __init__(self, file='dev.jsonl'):
        """
        file: a text file, each line is some sentences !!!!
        """
        super().__init__()
        # path = os.path.join(DATA_PATH, 'strategyqa', file)
        # self.data = read_jsonl(path) # data is a list now
        self.steps = 2
        self.stops = ['\nPassage:\n', None]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]['question']
    
    def test_output(self, idx: int, output: str):
        correct_answer = "the answer is " + re.search(r'#### (\d+)', self.data[idx]['answer']).group(1) if re.search(r'#### (\d+)', self.data[idx]['answer']) else None
        info = {"r": 1 if correct_answer in output else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    # @staticmethod
    # def cot_prompt_wrap(x: str, y:str='') -> str:
    #     return cot_prompt.format(input=x) + y
    
    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        if y:  # This means it's not the first step
            additional_prompt = "\n\nAbove is the previous thought. Build upon it to continue solving the question. You can modify the approach if needed. Answer to the question with with 'the answer is n', where n is a number."
            return cot_prompt.format(input=x) + y + additional_prompt
        else:  # This is the first step (step 0)
            return cot_prompt.format(input=x)

    # @staticmethod
    # def cot_prompt_wrap(x: str, y:str='') -> str:
    #     return cot_prompt.format(input=x) + y
    
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