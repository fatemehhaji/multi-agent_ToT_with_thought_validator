import os
import re
import json
import argparse
import random
from tqdm import tqdm
import openai
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent


def load_data(args):
    random_seed = 10
    random.seed(random_seed)
    data = [json.loads(line) for line in open(args.data_root)]
    random.shuffle(data)
    questions = {}
    answers = {}
    qids = []
    for i, item in enumerate(data):
        questions.update({str(i): item["question"]})
        # answers.update({str(i): 'True' if item["answer"] else 'False'})
        answers.update({str(i): re.search(r'#### (\d+)', item["answer"]).group(1) if re.search(r'#### (\d+)', item["answer"]) else ''})

        qids.append(str(i))
    
    # Limit the number of qids based on TEST_NUMBER if specified
    if args.test_number > 0:
        qids = qids[:args.test_number]
    
    print(f"Number of test problems: {len(qids)}\n")
    return questions, answers, qids

import re
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
    # Updated pattern to handle commas included in the number
    pattern = r"the answer is[^\d]*(\d+[\d,]*)"
    match = re.search(pattern, output_to_search, re.IGNORECASE)

    if match:
        # Extract and return the numeric part of the answer, replacing commas
        numeric_answer = match.group(1).replace(',', '')
        return numeric_answer
    
    # If no match found in the second part, search in the first part if it exists
    if len(indices) > 0 and indices[1] > 0:  # Ensure there is a first part to search
        first_part = output[:indices[1]]  # Search before the second occurrence
        match = re.search(pattern, first_part, re.IGNORECASE)
        if match:
            # Extract and return the numeric part of the answer, replacing commas
            numeric_answer = match.group(1).replace(',', '')
            return numeric_answer

    return ""  # If no match is found in either part


    
from time import sleep
def get_single_run_gpt(cot_prompt, prompt, args): 
    headers = {
            'Authorization': f'Bearer ' + args.api_key, 
            'Content-Type': 'application/json'
            }

    response = requests.post(args.api_url, 
                            headers=headers, 
                            json={
                                "model": 'gpt-3.5-turbo',
                                # "model": 'gpt-4',
                                # "model": 'gpt-4o-mini',
                                "messages": [
                                    {'role': 'user', 'content': prompt}, 
                                    ],
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                }).json()
    
    if 'error' in response.keys():
        print(response['error'])
    
    return response['choices'][0]['message']['content']


# ************************** gpt-3.5-turbo PROMPT (MATH) **************************
def create_verifier_prompt():
    verifier_prompt = f"""
    As a critical mathematical reasoning verifier, evaluate the following thought process, which builds upon previous steps to reach a final conclusion. Focus on:

    1. **Question Relevance**:
       - Ensure the entire reasoning process directly addresses the original question.
       - Check if the final answer actually solves what was asked.

    2. **Reasoning Progression**:
       - Assess logical flow and consistency, especially in final steps.
       - Verify mathematical operations' correctness and appropriateness.
       - Identify logical fallacies or unjustified leaps.

    3. **Factual Accuracy**:
       - Check accuracy and relevance of facts and numbers, particularly in final calculations.
       - Spot any misuse of mathematical concepts.

    4. **Completeness**:
       - Ensure all necessary aspects are addressed, particularly in concluding thoughts.
       - Identify significant omissions that could affect the result.

    5. **Critical Assessment**:
       - Actively seek potential errors or weak points.
       - Don't hesitate to invalidate reasoning if significant issues are found.

    Provide a holistic evaluation of the entire reasoning process, from start to finish. Conclude with "Reasoning is Valid" only if the entire process is relevant, logically sound, and error-free. Otherwise, conclude with "Reasoning is Invalid" and briefly explain why.
    """
    return verifier_prompt

# ************************** gpt-4o-mini PROMPT **************************
# def create_verifier_prompt():
#     verifier_prompt = f"""
#     You are an objective and fair verifier. Your role is to evaluate the overall logical soundness of a given reasoning path without generating or changing the final answer. Focus on the following:

#     1. **Evaluation of the Reasoning Process**:
#        - Is the reasoning generally logical and consistent?
#        - Does it address the main factors relevant to the question?
#        - Are there any major logical fallacies or significant unjustified leaps in reasoning?

#     2. **Factual Correctness**:
#        - Are the key facts presented in the reasoning generally accurate and relevant to the question?
#        - Is there any misuse or misinterpretation of critical information?

#     3. **Completeness of Analysis**:
#        - Does the reasoning consider the most important aspects of the problem?
#        - Are there any critical omissions in the thought process that would significantly alter the conclusion?

#     Provide your evaluation, focusing on the overall logical validity and completeness of the reasoning. Minor imperfections or slight omissions should not necessarily invalidate the entire reasoning. Do not generate or modify the final answer. 

#     Conclude with "Reasoning is Valid" if the reasoning is generally sound and addresses the main points of the question, even if it's not perfect. Only conclude with "Reasoning is Invalid" if there are major logical flaws, critical factual errors, or significant omissions that substantially impact the conclusion.
#     """
#     return verifier_prompt

# ************************** gpt-4o-mini PROMPT WITH TREE CONSIDERATION **************************
# def create_verifier_prompt():
#     verifier_prompt = f"""
#     You are an objective and fair verifier. Your role is to evaluate the overall logical soundness of a given reasoning path that may contain multiple levels of thought. This reasoning path is part of a Tree of Thoughts approach where earlier thoughts may be refined or contradicted by later ones. Focus on the following:

#     1. **Evaluation of the Reasoning Process**:
#        - Is the final reasoning generally logical and consistent?
#        - Does it address the main factors relevant to the question?
#        - Are there any major logical fallacies or significant unjustified leaps in the final conclusion?

#     2. **Progression of Thought**:
#        - How does the reasoning evolve from earlier to later thoughts?
#        - Are changes or contradictions between levels justified and explained?
#        - Does the final thought build upon or correct earlier thoughts in a logical manner?

#     3. **Factual Correctness**:
#        - Are the key facts presented in the final reasoning generally accurate and relevant to the question?
#        - Is there any misuse or misinterpretation of critical information in the final conclusion?

#     4. **Completeness of Analysis**:
#        - Does the final reasoning consider the most important aspects of the problem?
#        - Are there any critical omissions in the final thought process that would significantly alter the conclusion?

#     Provide your evaluation, focusing on the overall logical validity and completeness of the final reasoning. Consider how earlier thoughts contribute to or are refined by later ones. Minor imperfections, slight omissions, or initial thoughts that are later corrected should not necessarily invalidate the entire reasoning. Do not generate or modify the final answer.

#     Conclude with "Reasoning is Valid" if the final reasoning is generally sound and addresses the main points of the question, even if it's not perfect or if earlier thoughts were corrected. Only conclude with "Reasoning is Invalid" if there are major logical flaws, critical factual errors, or significant omissions in the final reasoning that substantially impact the conclusion.
#     """
#     return verifier_prompt
# ********************************************************************************************************

def verify_reasoning(reasoning, question, args):
    verifier_prompt = create_verifier_prompt()
    verify_prompt = f"{verifier_prompt}\n\nQuestion: {question}\n\nReasoning to verify:\n{reasoning}\n\nVerification:"
    
    verification = get_single_run_gpt(None, verify_prompt, args)
    return "reasoning is valid" in verification.lower(), verification


def get_most_probable_answer(verified_answers):
    true_count = verified_answers.count('True')
    false_count = verified_answers.count('False')
    total = true_count + false_count
    if total == 0:
        return random.choice(['True', 'False'])
    return 'True' if true_count >= false_count else 'False'


from methods.bfs import solve as tot_solve
from time import sleep

import random
from collections import Counter

def answer_review(task, question=None, depth=0, history='', args=None):
    if args.model == 'gpt':
        get_single_run = get_single_run_gpt
    else:
        raise NotImplementedError('undefined model')
    def get_reasoner_output(prompt):
        final_thought, output = tot_solve(args, task, prompt)
        return final_thought[0], output
    all_verified_answers = []
    all_answers = []
    while depth < 3:  # Max 3 rounds
        depth += 1
        history += f" ########## Round {depth} ########## \n"
        # Initial Reasoning and Verification
        r1_reasoning, r1_output = get_reasoner_output(question)
        r2_reasoning, r2_output = get_reasoner_output(question)
        r3_reasoning, r3_output = get_reasoner_output(question)
        history += f"***** Reasoner 1 output: *****\n{r1_output}\n"
        history += '*' * 50 + '\n'
        history += f"***** Reasoner 2 output: *****\n{r2_output}\n"
        history += '*' * 50 + '\n'
        history += f"***** Reasoner 3 output: ***** \n{r3_output}\n"
        history += '*' * 50 + '\n'
        verified_answers = []
        
        is_verified_r1, verification_r1 = verify_reasoning(r1_reasoning, question, args)
        final_answer_r1 = filter_output(r1_reasoning)
        all_answers.append(final_answer_r1)
        history += f"Reasoner 1: Answer: {final_answer_r1}, Verified: {is_verified_r1}\n"
        history += f"Verifier response:\n{verification_r1}\n"
        history += '*' * 50 + '\n'
        if is_verified_r1:
            verified_answers.append(final_answer_r1)
        
        
        is_verified_r2, verification_r2 = verify_reasoning(r2_reasoning, question, args)
        final_answer_r2 = filter_output(r2_reasoning)
        all_answers.append(final_answer_r2)
        history += f"Reasoner 2: Answer: {final_answer_r2}, Verified: {is_verified_r2}\n"
        history += f"Verifier response:\n{verification_r2}\n"
        history += '*' * 50 + '\n'
        if is_verified_r2:
            verified_answers.append(final_answer_r2)
        
        
        is_verified_r3, verification_r3 = verify_reasoning(r3_reasoning, question, args)
        final_answer_r3 = filter_output(r3_reasoning)
        all_answers.append(final_answer_r3)
        history += f"Reasoner 3: Answer: {final_answer_r3}, Verified: {is_verified_r3}\n"
        history += f"Verifier response:\n{verification_r3}\n"
        history += '*' * 50 + '\n'
        if is_verified_r3:
            verified_answers.append(final_answer_r3)
        all_verified_answers.extend(verified_answers)
        # Check for agreement among verified answers
        if len(verified_answers) >= 2:
            answer_counts = {}
            for answer in verified_answers:
                answer_counts[answer] = answer_counts.get(answer, 0) + 1
            
            for answer, count in answer_counts.items():
                if count >= 2:
                    history += f"At least two verified reasoners agree. Final Answer: {answer}\n\n"
                    return answer, history
        # If no agreement, continue to next round
        history += "No agreement among verified answers. Continuing to next round.\n\n"
    # After 3 rounds or if no agreement, return the most frequent answer from all verified answers
    if all_verified_answers:
        final_answer = max(set(all_verified_answers), key=all_verified_answers.count)
        history += f"Maximum rounds reached or no agreement. Choosing most frequent answer: {final_answer}\n\n"
    else:
        # If no verified answers, return the most repeated from all_answers
        answer_counts = Counter(all_answers)
        most_common = answer_counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # If all answers repeat the same number of times, return a random one
            final_answer = random.choice(all_answers)
            history += f"No verified answers and all answers repeated equally. Randomly selected answer: {final_answer}\n\n"
        else:
            final_answer = most_common[0][0]
            history += f"No verified answers. Choosing most frequent answer from all answers: {final_answer}\n\n"
    
    return final_answer, history


def get_single_result(qid, question, args):
    if args.method == 'ours':
        answer, output = answer_review(task=task, question=question, depth=0, history='', args=args)
        
    return qid, answer, output


def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    else:
        return random.choice(range(len(choices)))


def get_result_file(args):
    result_file = "{}/{}/{}_{}_seed_{}.json".format(args.output_root, args.model, args.label, args.test_split, args.seed)

    os.makedirs(os.path.dirname(args.output_root), exist_ok=True)

    return result_file


def save_results(result_file, acc, correct, count, args, results, outputs):
    data = {}
    data['acc'] = acc
    data['correct'] = correct
    data['count'] = count
    data['args'] = vars(args)
    data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='')
    parser.add_argument('--output_root', type=str, default='./results')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    # user options
    parser.add_argument('--label', type=str, default='exp0')
    parser.add_argument('--test_split', type=str, default='val', choices=['test', 'val', 'minival', 'train'])
    parser.add_argument('--txt_only', default=False, action='store_true')
    parser.add_argument('--subset', default=False, action='store_true')
    parser.add_argument('--test_number', type=int, default=10, help='GPT-3 is expensive. -1 for whole val/test set')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--save_every', type=int, default=10, help='Save the result with every n examples.')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_format',
                        type=str,
                        default='CQM-A',
                        choices=[
                            'CQM-A', 'CQM-LA', 'CQM-EA', 'CQM-LEA', 'CQM-ELA', 'CQM-AL', 'CQM-AE', 'CQM-ALE', 'QCM-A',
                            'QCM-LA', 'QCM-EA', 'QCM-LEA', 'QCM-ELA', 'QCM-AL', 'QCM-AE', 'QCM-ALE', 'QCML-A', 'QCME-A',
                            'QCMLE-A', 'QCLM-A', 'QCEM-A', 'QCLEM-A', 'QCLM-AE'
                        ],
                        help='prompt format template')
    parser.add_argument('--shot_number', type=int, default=3, help='Number of n-shot training examples.')
    parser.add_argument('--shot_qids', nargs='+', type=int, help='Question indexes of shot examples')
    parser.add_argument('--seed', type=int, default=10, help='random seed')
    # GPT-3 settings
    parser.add_argument('--api_key', type=str, help='OpenAI API key')
    parser.add_argument('--api_url', type=str, help='OpenAI API URL')
    parser.add_argument('--engine', type=str, default='text-davinci-002')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--max_tokens',
                        type=int,
                        default=2000,
                        help='The maximum number of tokens allowed for the generated answer.')
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--method', type=str, default='reviewer')

    parser.add_argument('--task', type=str, default='gsm8k', help='Specify the task.')
    parser.add_argument('--method_generate', type=str, default='sample', help='Method for generation (e.g., sample).')
    parser.add_argument('--method_evaluate', type=str, default='vote', help='Method for evaluation (e.g., vote).')
    parser.add_argument('--method_select', type=str, default='greedy', help='Method for selection (e.g., greedy).')
    parser.add_argument('--n_generate_sample', type=int, default=5, help='Number of samples to generate.')
    parser.add_argument('--n_evaluate_sample', type=int, default=5, help='Number of samples to evaluate.')
    parser.add_argument('--n_select_sample', type=int, default=1, help='Number of samples to select.')
    parser.add_argument('--prompt_sample', type=str, default='cot', help='Prompt sample type (e.g., cot for Chain of Thought).')
    parser.add_argument('--backend', type=str, default='gpt-3.5-turbo', help='')

    from tasks import get_task
    global task
    args = parser.parse_args()
    task = get_task(args.task)

    return args


if __name__ == '__main__':
    args = parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    np.random.seed(args.seed)

    sents, labels, qids = load_data(args)  # probelms, test question ids, shot example ids

    if args.subset:
        qids = qids[::20]

    result_file = get_result_file(args)
    print(result_file)

    # load the check point
    if os.path.exists(result_file):
        print("# The result file exists! We will load the check point!!!")
        check_point = json.load(open(result_file))
        acc = check_point['acc']
        correct = check_point['correct']
        results = check_point['results']
        outputs = check_point['outputs']
        print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
    else:
        correct = 0
        results = {}
        outputs = {}

    n = 0
    for i, qid in enumerate(qids):
        if qid in results:
            continue
        
        # Get the result sequentially, without using futures
        qid, prediction, output = get_single_result(qid, sents[str(i)], args)

        n += 1

        label = labels[qid]

        results[qid] = prediction
        outputs[qid] = output
        if prediction == label:
            correct += 1

        acc = correct / len(results) * 100

        if True:  # args.debug or i < 50:
            print('\n\n\n', "###" * 15, 'START', "###" * 15)
            print('# Num: ', qid)
            print("# full output:", output)
            print("# labeled answer:", label)
            print("# predicted answer:", prediction)
            print("# Correct:", prediction == label)
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%")
            print("###" * 15, 'END', "###" * 15, '\n\n\n')

        if (n + 1) % args.save_every == 0 or (n + 1) == len(qids):
            print(f"{len(results)}/{len(qids)}, correct: {correct}, acc: {round(acc, 2)}%, saving to {result_file}")
            save_results(result_file, acc, correct, n + 1, args, results, outputs)

    # Final save at the end
    save_results(result_file, acc, correct, len(results), args, results, outputs)