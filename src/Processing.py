import json
import tiktoken
from random import shuffle
import re
import pandas as pd

class Processing:

    def __init__(self):
        pass

    # Parses the value file to return the dict of values in the appropriate level
    # @param values_file The file containing the values JSON
    # @return dict with values in the appropriate level
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
    def get_value_name_list(values):
        shuffle(values)
        res = {}
        level_1 = set()
        level_2 = set()
        level_3 = set()
        level_4a = set()
        res["Level 1"] = level_1
        res["Level 2"] = level_2
        res["Level 3"] = level_3
        res["Level 4A"] = level_4a
        res["Level 4B"] = {'Growth, Anxiety-free', 'Self-protection, Anxiety avoidance'}
        for value in values:
            res["Level 1"].add(value['name'])
            res["Level 2"].add(value['level2'])
            res["Level 3"].update(value['level3'])
            res["Level 4A"].update(value['level4a'])
        return res

    # Converts the value list to a string form for prompting
    @staticmethod
    def stringify_values_for_prompt(values):
        shuffle(values)
        res = Processing.get_value_name_list(values)

        stringified = ""
        for key, value in res.items():
            if key == 'Level 4B':
                stringified += key + "values: \"Growth, Anxiety-free\" and \"Self-protection, Anxiety avoidance\". " \
                                     "Note that these are two values in total, not four. "
            else:
                stringified += key + " values: " + ", ".join(value) + "\n"

        return stringified

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
