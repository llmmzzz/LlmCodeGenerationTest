import importlib.util
import json
import subprocess

import google.generativeai as genai
import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer
from mysrc.myio_utils import Toolls
from mysrc.myevaluation import run_script_of_bleu
from mysrc.myevaluation import run_script_of_rouge

# llm list for chosen
llm_list = ['gemini-pro']


def llm_operation(dataset_task_id, dataset2_task_id):
    global llm_list
    # print llm can choose
    print(llm_list)
    print("input 1 to choose our llm,input 2 to choose by yourself")
    judge = input()
    if judge == '1':
        print("input llm name")
        input_llm = input()
        if input_llm == "gemini-pro":
            gemini_pro_operation(dataset_task_id, dataset2_task_id)
    if judge == '2':
        print("input configfile path")
        config_file = input()
        # install_package(package)
        # import_module(module_name)
        config_data = config_extract(config_file)
        print(config_data)
        config_operation(config_data, dataset_task_id, dataset2_task_id)


def gemini_pro_operation(dataset_task_id, dataset2_task_id):
    output_dict = dict()
    output_dict2 = dict()
    output_dict3 = dict()
    genai.configure(api_key="AIzaSyAKqQbekkIONUAw4Qgprjm97chsTfvL9Zg")
    model = genai.GenerativeModel('gemini-pro')
    for task_id in dataset_task_id:
        response = model.generate_content(dataset_task_id[task_id]['prompt'], stream=True)
        response.resolve()
        print(response.text)
        output_dict[task_id] = {'task_id': task_id,
                                'completion': response.text}
        print(output_dict[task_id])
        output_dict2[task_id] = {'prompt': dataset_task_id[task_id]['prompt'],
                                 'samples': [response.text]}
        score_bleu = run_script_of_bleu(dataset_task_id[task_id]['test'], response.text)
        score_str_bleu = str(score_bleu)
        print(dataset_task_id[task_id]['task_id'] + " score_bleu= " + score_str_bleu)
        score_rouge = run_script_of_rouge(dataset_task_id[task_id]['test'], response.text)
        print(dataset_task_id[task_id]['task_id'] + " score_rouge= ")
        print(score_rouge)
    for task_id in dataset2_task_id:
        response = model.generate_content(dataset2_task_id[task_id]['prompt'] + "please generate 1 test case to test the accuracy of the code you generated and just give me the test case"
                                          , stream=True)
        response.resolve()
        print(response.text)
        output_dict3[task_id] = {'prompt': dataset2_task_id[task_id]['prompt'],
                                 'samples': [response.text]}
    Toolls.output_to_jsonl(output_dict, "data/generated_data/data_for_gemini-pro/geminipro_samples_for_MBPP.jsonl")
    Toolls.output_to_jsonl(output_dict2, "data/generated_data/data_for_gemini-pro/MBPP_geminipro_code_solution.jsonl")
    Toolls.output_to_jsonl(output_dict3, "data/generated_data/data_for_gemini-pro/MBPP_geminipro_test_case.jsonl")


# setup python package
def install_package(package):
    subprocess.check_call(['pip', 'install', package])


# import module
def import_module(module_name):
    spec = importlib.util.find_spec(module_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# load config file
def read_json_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config_data = json.load(file)
    return config_data


# extract config file
def config_extract(config_file):
    config_data = read_json_config(config_file)
    return config_data


# handle json file data
def config_operation(config_data, dataset_task_id, dataset2_task_id):
    handled_data = dict()
    output_dict = dict()
    output_dict2 = dict()
    output_dict3 = dict()
    for key in config_data.keys():
        if key == "model_name":
            print(config_data[key])
            handled_data[key] = config_data[key]
        if key == "max_length":
            handled_data[key] = int(config_data[key])
        if key == "temperature":
            handled_data[key] = float(config_data[key])
        if key == "do_sample":
            handled_data[key] = bool(config_data[key])
        if key == "top_k":
            handled_data[key] = int(config_data[key])
        if key == "top_p":
            handled_data[key] = float(config_data[key])
        if key == "no_repeat_ngram_size":
            handled_data[key] = int(config_data[key])
        if key == "num_return_sequences":
            handled_data[key] = int(config_data[key])
    model = AutoModelForCausalLM.from_pretrained(handled_data['model_name'])
    # model.eval()
    # if torch.cuda.is_available():
    #     model.to(torch.device("cuda:2"))
    tokenizer = AutoTokenizer.from_pretrained(handled_data['model_name'])
    for task_id in dataset_task_id:
        input_text = dataset_task_id[task_id]['prompt']
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids=input_ids,
                                max_length=handled_data['max_length'],
                                temperature=handled_data['temperature'],
                                do_sample=handled_data['do_sample'],
                                top_k=handled_data['top_k'],
                                top_p=handled_data['top_p'],
                                no_repeat_ngram_size=handled_data['no_repeat_ngram_size'],
                                num_return_sequences=handled_data['num_return_sequences'],
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=50256,
                                attention_mask=input_ids,
                                )
        generated_text = tokenizer.decode(output[0], skip_specials_tokens=True)
        output_dict[task_id] = {'task_id': task_id,
                                'completion': generated_text}
        output_dict2[task_id] = {'prompt': dataset_task_id[task_id]['prompt'],
                                 'samples': [generated_text]}
        print("Generated text:", generated_text)
        score_bleu = run_script_of_bleu(dataset_task_id[task_id]['test'], generated_text)
        score_str_bleu = str(score_bleu)
        print(dataset_task_id[task_id]['task_id'] + " score_bleu= " + score_str_bleu)
        score_rouge = run_script_of_rouge(dataset_task_id[task_id]['test'], generated_text)
        print(dataset_task_id[task_id]['task_id'] + " score_rouge= ")
        print(score_rouge)
    for task_id in dataset2_task_id:
        input_text = dataset2_task_id[task_id]['prompt'] + "please generate 1 test case for this task and just give me test case"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        output = model.generate(input_ids=input_ids,
                                max_length=handled_data['max_length'],
                                temperature=handled_data['temperature'],
                                do_sample=handled_data['do_sample'],
                                top_k=handled_data['top_k'],
                                top_p=handled_data['top_p'],
                                no_repeat_ngram_size=handled_data['no_repeat_ngram_size'],
                                num_return_sequences=handled_data['num_return_sequences'],
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=50256,
                                attention_mask=input_ids
                                )
        generated_text = tokenizer.decode(output[0], skip_specials_tokens=True)
        output_dict3[task_id] = {'prompt': dataset2_task_id[task_id]['prompt'],
                                 'samples': [generated_text]}
    Toolls.output_to_jsonl(output_dict, "data/generated_data/data_for_gemini-pro/geminipro_samples_for_HumanEval.jsonl")
    Toolls.output_to_jsonl(output_dict2, "data/generated_data/data_for_gemini-pro/HumanEval_geminipro_code_solution.jsonl")
    Toolls.output_to_jsonl(output_dict3, "data/generated_data/data_for_gemini-pro/HumanEval_geminipro_test_case.jsonl")
