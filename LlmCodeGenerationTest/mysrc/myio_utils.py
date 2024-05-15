import json


class Toolls:
    # from jsonl get json in 'dict' format
    @staticmethod
    def load_jsonl(file_path):
        json_objects = []
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                json_objects.append(json.loads(line.strip()))
        return json_objects

    # standardize dataset
    @staticmethod
    def standardize_dataset(file_path):
        # json_objects in List[dict] format
        json_objects = Toolls.load_jsonl(file_path)
        # get keys
        key_list = list(json_objects[0].keys())
        print(key_list)

        # get dataset input choice
        inputs_task_id = []
        inputs_prompt = []
        inputs_canonical_solution = []
        inputs_test = []
        inputs_entry_point = []
        print("join prompt key,input done to end")
        while True:
            user_input = input()
            if user_input == 'done':
                break
            inputs_prompt.append(user_input)
        print("join task_id key,input done to end")
        while True:
            user_input = input()
            if user_input == 'done':
                break
            inputs_task_id.append(user_input)
        print("join canonical_solution key,input done to end")
        while True:
            user_input = input()
            if user_input == 'done':
                break
            inputs_canonical_solution.append(user_input)
        print("join test key,input done to end")
        while True:
            user_input = input()
            if user_input == 'done':
                break
            inputs_test.append(user_input)
        print("join entry_point key,input done to end")
        while True:
            user_input = input()
            if user_input == 'done':
                break
            inputs_entry_point.append(user_input)
        # standardize dataset by input choice
        dataset_prompt = dict()
        dataset_task_id = dict()
        for line in json_objects:
            prompt_str = str()
            task_id_str = str()
            canonical_solution_str = str()
            test_str = str()
            entry_point_str = str()
            for key in line.keys():
                # this key is chosen to combine prompt
                if key in inputs_prompt:
                    prompt_str += line[key]
                # this key is chosen to combine response
                if key in inputs_task_id:
                    task_id_str += line[key]
                if key in inputs_canonical_solution:
                    canonical_solution_str += line[key]
                if key in inputs_test:
                    test_str += line[key]
                if key in inputs_entry_point:
                    entry_point_str += line[key]
            dataset_prompt[prompt_str] = {'task_id': task_id_str,
                                          'prompt': prompt_str,
                                          'canonical_solution': canonical_solution_str,
                                          'test': test_str,
                                          'entry_point': entry_point_str}
            dataset_task_id[task_id_str] = {'task_id': task_id_str,
                                            'prompt': prompt_str,
                                            'canonical_solution': canonical_solution_str,
                                            'test': test_str,
                                            'entry_point': entry_point_str}
        return dataset_prompt, dataset_task_id

    @staticmethod
    def output_to_jsonl(output_dict, output_file):
        with open(output_file, 'w') as f:
            for key, value in output_dict.items():
                value_json = json.dumps(value)
                f.write(value_json + '\n')
