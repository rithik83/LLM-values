import json
import tiktoken
from random import shuffle
import re
import pandas as pd

class Processing:

    def __init__(self):
        pass

    # Parses the value file to return the dict of values in the appropriate level
    @staticmethod
    def parse_value_file(values_file):
        f = open(values_file)
        data = json.load(f)

        values = data['values']

        return values

    @staticmethod
    def num_tokens_from_string(string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    # Returns a string of the value names from the list
    @staticmethod
    def value_names(value_list):
        shuffle(value_list)
        res = ""
        for value in value_list:
            res += value['name'] + ', '

        return res

    # Returns a list of the value names from the list of values
    @staticmethod
    def get_values_dict(values):
        shuffle(values)
        res = {}
        level_1 = []
        level_2 = set()
        level_3 = set()
        level_4a = set()
        res["Level 1"] = level_1
        res["Level 2"] = level_2
        res["Level 3"] = level_3
        res["Level 4A"] = level_4a
        res["Level 4B"] = {'Growth, Anxiety-free', 'Self-protection, Anxiety avoidance'}
        for value in values:
            res["Level 1"].append({'name': value['name'], 'descriptions': value['descriptions']})
            res["Level 2"].add(value['level2'])
            res["Level 3"].update(value['level3'])
            res["Level 4A"].update(value['level4a'])
        return res

    # Converts the value list to a string form for prompting
    @staticmethod
    def stringify_values_for_prompt(values, level):
        res = Processing.get_values_dict(values)
        values = res[level]

        if level == 'Level 1':
            string_for_prompt = Processing.stringify_level1_values(values)
        elif level == 'Level 4B':
            string_for_prompt = "\"Growth, Anxiety-free\" and \"Self-protection, Anxiety avoidance\". Note that " \
                                "(Growth, Anxiety-free) is altogether one value, and likewise (Self-protection, " \
                                "Anxiety avoidance) is also altogether one value."
        else:
            string_for_prompt = ", ".join(values) + "\n"

        return string_for_prompt

    @staticmethod
    def select_examples_fewshot(args, values_dict, level, num_examples, random_state):
        target_frequency = num_examples
        subset = pd.DataFrame(columns=args.columns)

        value_list = values_dict[level]
        if level == "Level 1":
            value_list = [value["name"] for value in value_list]

        # Step 1: Creating a mini-dataframe of subset arguments, that represents every value from the level at least "num_examples" times
        for value in value_list:
            value_subset = args[args[value] == 1]
            while len(value_subset) < target_frequency:
                value_subset = pd.concat([value_subset, args[args[value] == 1]], ignore_index=True)
            subset = pd.concat([subset, value_subset.sample(frac=1, random_state=random_state)[:int(target_frequency)]], ignore_index=True)

        # Sampling "num_examples" datapoints from the subset dataframe
        sampled_arguments = subset.sample(num_examples, random_state=random_state)

        # For the values that do not occur in sampled_arguments, adds at least one more example from the subset which includes the value
        for value in value_list:
            num_args_with_value = sampled_arguments[sampled_arguments[value] == 1]
            if len(num_args_with_value) == 0:
                value_subset = subset[subset[value] == 1].sample(1, random_state=random_state)
                sampled_arguments = pd.concat([sampled_arguments, value_subset], ignore_index=True)

        sampled_arguments = sampled_arguments.reset_index(drop=True)
        return sampled_arguments

    @staticmethod
    def generate_example_string_fewshot(examples, values_dict, level):
        prompt_string = ""
        values_to_consider = []

        for value in values_dict[level]:
            if level == "Level 1":
                values_to_consider.append(value["name"])
            else:
                values_to_consider.append(value)

        for index in range(len(examples)):
            policy = examples.at[index, 'Conclusion']
            opinion = examples.at[index, 'Premise']
            stance = examples.at[index, 'Stance']

            true_values = []

            for value in values_to_consider:
                if examples.at[index, value] == 1:
                    true_values.append(value)

            true_values_string = '; '.join(true_values)

            prompt_string += f"Q: Someone is {stance} the idea {policy}, arguing that {opinion}. What are the " \
                             f"human values that motivate their opinion?\nA: {true_values_string}\n"

        return prompt_string

    @staticmethod
    def individual_example_prompt_string_cot(examples_list):
        template = ""

        for example in examples_list:
            template += (
                f"Q: Someone is {example['stance']} the idea {example['policy']}, arguing that {example['opinion']}. "
                f"What are the human values that motivate their opinion?\n"
            )

            for value in example['values']:
                template += "Reasoning:\nValue: " + value + "\n"

        return template

    @staticmethod
    def parse_values_generated(llm_result):
        values = []

        for line in llm_result.split('\n'):
            if line.startswith('Value:'):
                value = line.replace('Value: ', '', 1).strip()
                values.append(value)

        return values

    @staticmethod
    def stringify_level1_values(level1_set):
        shuffle(level1_set)
        string_l1_values = ", ".join([value['name'] for value in level1_set])
        return string_l1_values

    @staticmethod
    def sample_rows(group, k):
        return group.sample(k)

    @staticmethod
    def sample_arguments(arguments, k):
        return arguments.groupby('Conclusion', group_keys=False).apply(Processing.sample_rows, k=k).reset_index()


    @staticmethod
    def extract_values_from_result(llm_answer):
        value_pattern = re.compile(r'VALUE:\s*(.*?)\s*Justification:', re.DOTALL)
        value_matches = value_pattern.findall(llm_answer)

        return value_matches

    @staticmethod
    def extract_values_alternate(llm_answer):
        words_list = []
        lines = llm_answer.split('\n')
        for line in lines:
            word = line.split(":")[1].strip()
            words_list.append(word)
        return words_list

    @staticmethod
    def extract_values_real(llm_answer, with_cot):
        values_dict = {}

        lines = llm_answer.strip().split('\n')

        for line in lines:
            level, sep, values = line.partition('Values:')
            level = level.strip()
            values = [value.strip() for value in values.split(';')] if values else []
            values_dict[level] = values

        return values_dict

    @staticmethod
    def merge_with_labels(arguments, labels):
        return pd.merge(arguments, labels, on='Argument ID', how='inner')

    @staticmethod
    def generate_example_string(train_arguments, conclusion, values_dict, level, num_examples):

        examples_in_favor = train_arguments[(train_arguments["Stance"] == "in favor of")].sample(num_examples)
        examples_against = train_arguments[(train_arguments["Stance"] == "against")].sample(num_examples)

        examples = examples_in_favor + examples_against
        examples.reset_index(inplace=True)
        example_string = ""

        for index in range(len(examples)):
            policy = examples.at[index, 'Conclusion']
            stance = examples.at[index, 'Stance']
            opinion = examples.at[index, 'Premise']

            example_string += f"Q: Someone is {stance} the policy {policy}, arguing that {opinion}. What are the human values that motivate their opinion?\n"

            if level == "Level 1":
                list_of_l1_values = values_dict[level]
                values = [value['name'] for value in list_of_l1_values]
            else:
                values = [value for value in values_dict[level]]

            # Select subset of columns
            subset_df = examples.loc[index, values]

            # Find columns with value 1
            columns_with_1 = subset_df[subset_df == 1].index.tolist()

            example_string += "A: " + "; ".join(columns_with_1) + "\n"

        return example_string

