from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from Processing import Processing
from Evaluation import Evaluation
from functools import reduce
import time


class Prompting:

    def __init__(self):
        pass

    # Performs an individual prompt to the specified LLM, gathering its identified values only (without CoT)
    # @param policy The conclusion or policy towards which the opinion is directed
    # @param opinion The opinion towards the policy
    # @param stance The stance of the opinion in relation to the policy
    # @param llm The Large Language Model used
    # @param values A string with each value and its description
    # @return A string with each of the values, as well as their justifications and relation to the opinion
    @staticmethod
    def zero_shot_prompt(policy, opinion, stance, llm, values, level):
        template = (
            "Someone is {stance} the idea {policy}, arguing that {opinion}.\n"
            "You are presented with a multi-level categorization of human values, and tasked with selecting the "
            "appropriate level 1, 2, 3, 4a and 4b values directly or indirectly implied by the argument made.\n"
            "The list of human values to choose from:\n"
            "{values_string}"
            "Represent your answer as below:\n"
            "### ANSWER FORMAT GUIDE ###\n"
            "Level 1 Values: <semicolon separated list of level 1 values>"
            "Level 2 Values: <semicolon separated list of level 2 values>"
            "Level 3 Values: <semicolon separated list of level 3 values>"
            "Level 4A Values: <semicolon separated list of level 4A values>"
            "Level 4B Values: <semicolon separated list of level 4B values>"
        )

        values_string = Processing.stringify_values_for_prompt(values)

        prompt = PromptTemplate(template=template, input_variables=['opinion', 'policy', 'stance', 'values_string'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "opinion": opinion,
            "policy": policy,
            "stance": stance,
            "values_string": values_string
        })

        return answer

    @staticmethod
    def prompt_for_justifications(example_arguments, llm, values, labels_level1, labels_level2, labels_level3,
                                  labels_level4a, labels_level4b):
        arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                           [example_arguments, labels_level1, labels_level2, labels_level3, labels_level4a,
                            labels_level4b])

        value_name_list = Processing.get_value_name_list(values)
        level_1_values = value_name_list["Level 1"]
        level_2_values = value_name_list["Level 2"]
        level_3_values = value_name_list["Level 3"]
        level_4a_values = value_name_list["Level 4A"]
        level_4b_values = value_name_list["Level 4B"]

        example_string = ""

        for index in range(len(arguments)):
            policy = arguments.at[index, 'Conclusion']
            stance = arguments.at[index, 'Stance']
            opinion = arguments.at[index, 'Premise']

            level_1_values = [value for value in level_1_values if arguments.at[index, value] == 1]
            level_2_values = [value for value in level_2_values if arguments.at[index, value] == 1]
            level_3_values = [value for value in level_3_values if arguments.at[index, value] == 1]
            level_4a_values = [value for value in level_4a_values if arguments.at[index, value] == 1]
            level_4b_values = [value for value in level_4b_values if arguments.at[index, value] == 1]


            example_string += f"Example {index+1}: Someone is {stance} the idea {policy}, arguing that {opinion}.\n " \
                              "The moral values they directly or indirectly imply are as follows:\n Level 1 Values: " \
                              ", ".join(level_1_values) + "\n" + "Level 2 Values: " + ", ".join(level_2_values) + \
                              "\n" + "Level 3 Values: " + ", ".join(level_3_values) + "\nLevel 4A Values: " + \
                              ", ".join(level_4a_values) + "\nLevel 4B Values: " + ", ".join(level_4b_values) + "\n"

        template = ("Following are examples of arguments made by people, and the moral values that are directly or "
                    "indirectly implied by them:\n" + example_string + "\n"
                    "Your task is to come up with justifications for why each of the values have been chosen. "
                    "Repeat the below answer format for all the examples\n"
                    "### ANSWER FORMAT GUIDE ###\n"
                    "Example 1:\n"
                    "Justification for Level 1 Values: <Your justification for all the level 1 values>"
                    "Justification for Level 2 Values: <Your justification for all the level 2 values>"
                    "Justification for Level 3 Values: <Your justification for all the level 3 values>"
                    "Justification for Level 4A Values: <Your justification for all the level 4A values>"
                    "Justification for Level 4B Values: <Your justification for all the level 4B values>")

        prompt = PromptTemplate(template = template, input_variables=[])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({})

        return answer

    # Performs the zero-shot prompting sequence for all the arguments given in as input.
    # @param arguments The arguments to be input to the LLM
    # @param values The moral values to be considered
    # @param llm The Large Language Model used
    # @param value_labels the true labels
    # @param verbose Boolean to indicate if print statements are needed
    # @param sleep_time Time to wait between subsequent prompts (for APIs with rate limits)
    # @param with_cot Boolean to indicate whether to use zero-shot with CoT or without
    # @return Predicted Labels and the arguments with the true labels
    @staticmethod
    def zero_shot_sequence(arguments, values, llm, labels_level1, labels_level2, labels_level3, labels_level4a,
                           labels_level4b, verbose, with_cot):

        arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                           [arguments, labels_level1, labels_level2, labels_level3, labels_level4a, labels_level4b])

        value_name_list = Processing.get_value_name_list(values)
        level_1_values = value_name_list["Level 1"]
        level_2_values = value_name_list["Level 2"]
        level_3_values = value_name_list["Level 3"]
        level_4a_values = value_name_list["Level 4A"]
        level_4b_values = value_name_list["Level 4B"]

        predicted_labels = arguments.copy()
        combined_list = []
        for s in [level_1_values, level_2_values, level_3_values, level_4a_values, level_4b_values]:
            combined_list.extend(s)
        predicted_labels[combined_list] = 0

        for index in range(len(arguments)):
            policy = arguments.at[index, 'Conclusion']
            stance = arguments.at[index, 'Stance']
            opinion = arguments.at[index, 'Premise']

            if with_cot:
                llm_result = Prompting.zero_shot_cot_prompt(policy, opinion, stance, llm, values)
            else:
                llm_result = Prompting.zero_shot_prompt(policy, opinion, stance, llm, values)

            predicted_values = Processing.extract_values_real(llm_result, with_cot)

            if verbose is True:
                print(f"index: {index}, policy: {policy}, opinion: {opinion}", "\n")
                print(predicted_values, "\n")

            existing_values = level_1_values.union(level_2_values, level_3_values, level_4a_values, level_4b_values)\
                              & set().union(*predicted_values.values())

            predicted_labels.iloc[index, predicted_labels.columns.isin(existing_values)] = 1

        predicted_labels = predicted_labels[combined_list]
        true_labels = arguments[combined_list]

        return predicted_labels, true_labels