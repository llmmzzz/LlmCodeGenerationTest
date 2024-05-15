from mysrc.myio_utils import Toolls
from mysrc.llms import llm_operation
from mysrc.myevaluation import run_script_of_pass_k

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import json

from src.postprocess import PostProcessor
from src.execution import evaluate_with_test_code, evaluate_with_test_cases
from src.io_utils import Tools
from src.agreement import DataManager, DualAgreement
from src.evaluation import pass_at_K, get_result_of_sorted_solutions

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def codet_run():
    print("input source_path_for_solution")
    source_path_for_solution = input()
    print("input predict_path_for_solution")
    predict_path_for_solution = input()
    print("input source_path_for_test")
    source_path_for_test = input()
    print("input predict_path_for_test")
    predict_path_for_test = input()
    print("input cache_dir_for_HumanEval")
    cache_dir = input()
    print("input timeout")
    timeout = input()
    print("input test_case_limit")
    test_case_limit = input()

    handled_solutions, task_count = PostProcessor.map_task_id_for_solution(predict_path_for_solution,
                                                                           source_path_for_solution)
    handled_test_cases = PostProcessor.map_task_id_for_test_case(predict_path_for_test, source_path_for_test)

    ground_truth_exec_result = evaluate_with_test_code(handled_solutions, timeout=float(timeout))
    dual_exec_result = evaluate_with_test_cases(handled_solutions, handled_test_cases, timeout=float(timeout),
                                                limit=int(test_case_limit))

    Tools.dump_pickle(os.path.join(cache_dir, 'ground_truth_exec_result.pkl'), ground_truth_exec_result)
    Tools.dump_pickle(os.path.join(cache_dir, 'dual_exec_result.pkl'), dual_exec_result)

    data_manager = DataManager(dual_exec_result, handled_solutions, handled_test_cases, int(test_case_limit))
    set_consistency = DualAgreement(data_manager)
    ranked_result = set_consistency.get_sorted_solutions_without_iter()
    logger.info('pass rates of ranked solutions')
    get_result_of_sorted_solutions(ground_truth_exec_result, ranked_result)
    logger.info('pass rates of random solutions')
    pass_at_K(ground_truth_exec_result)


if __name__ == '__main__':
    # data = {
    #     'model_name': 'codellama/CodeLlama-7b-hf',
    #     'max_length': '800',
    #     'num_return_sequences': '1',
    #     'temperature': '0.1',
    #     'do_sample': 'True',
    #     'top_k': '50',
    #     'top_p': '1.0',
    #     'no_repeat_ngram_size': '0'
    # }
    # with open('data/configCodeLlama-7b-hf.json', 'w') as f:
    #     json.dump(data, f)
    #
    # config_path = input()
    # config_data = config_extract(config_path)
    # handled_data = config_run(config_data)
    # print(handled_data)

    # get dataset input (.jsonl)
    print("input code_generation_and_test or data_test to choose what to do")
    choose_what_to_do = input()
    if choose_what_to_do == "a":
        codet_run()
    if choose_what_to_do == "c":
        problems_json = input()
        samples_json = input()
        run_script_of_pass_k(problems_json, samples_json)
    if choose_what_to_do == "b":
        json_dict = dict()
        count = 0
        print("input the file path")
        file_path = input()
        json_objects = Toolls.load_jsonl(file_path)
        for line in json_objects:
            prompt_str = str()
            samples_str = str()
            for key in line.keys():
                if key == "prompt":
                    prompt_str += line[key]
                if key == "samples":
                    samples_str += line[key][0]
                    print(line[key][0])
                    print(samples_str)
            task_id = "HumanEval/" + str(count)
            json_dict[task_id] = {'task_id': task_id,
                                   'completion': samples_str}
            count += 1
        Toolls.output_to_jsonl(json_dict, "data/generated_data/data_for_incoder6B/incoder6B_samples_for_HumanEval.jsonl")

    if choose_what_to_do == "code_generation_and_test":
        print("input 1 to choose dataset by yourself or 2 to choose human-eval")
        choose_dataset_input = input()
        dataset_input = str()
        dataset2_input = str()
        if choose_dataset_input == "1":
            print("input dataset for solution path")
            dataset_input = input()
            print("if get pass@k,dataset must be in format")
            print("input dataset for test case path")
            dataset2_input = input()
        if choose_dataset_input == "2":
            dataset_input = "data/HumanEval_for_code_generation.jsonl"
            dataset2_input = "data/HumanEval_for_test_case_generation.jsonl"
        # get List[dict] of jsonl
        dataset_prompt, dataset_task_id = Toolls.standardize_dataset(dataset_input)
        dataset2_prompt, dataset2_task_id = Toolls.standardize_dataset(dataset2_input)
        llm_operation(dataset_task_id, dataset2_task_id)
        print("input your samples jsonl path")
        samples_json = input()
        run_script_of_pass_k(dataset_input, samples_json)
        codet_run()
    if choose_what_to_do == "data_test":
        print("we provide metric CodeT, rouge, pass@k, bleu")
        print("for every task, rouge and bleu is necessary")
        print("to choose CodeT or pass@k, finish the README first")
        print("input your problem json path")
        problem_json = input()
        print("input your samples jsonl path")
        samples_json = input()
        run_script_of_pass_k(problem_json, samples_json)
        codet_run()
