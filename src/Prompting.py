from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from Processing import Processing
from functools import reduce


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
            "You are an expert in debate, argumentation and moral inclinations."
            "Someone is {stance} the statement {policy}, arguing that {opinion}."
            "Your task is to determine the human values that motivate and form the foundation behind the opinion stated."
            "Choose values from the below list, and do not make up your own values.\n"
            "{values_string}\n"
            "Represent your answer as a semicolon separated list of values. Be as selective and precise as possible."
            "Only choose the human values that very clearly are associated with and align with the premise."
            "Do not choose irrelevant or barely related values. Be precise and selective."
        )

        values_string = Processing.stringify_values_for_prompt(values, level)

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
    def few_shot_prompt(policy, opinion, stance, llm, values, level, example_string):
        template = (
            "### VALUE LIST ###\n"
            "{values_string}\n"
            "{example_string}\n"
            "Q: Someone is {stance} the idea {policy}, arguing that {opinion}. What are the human values that motivate their opinion?\n"
            "A: <your selection of values (separated by semicolon) based on the thought process established in past examples>"
        )

        values_string = Processing.stringify_values_for_prompt(values, level)

        prompt = PromptTemplate(template=template, input_variables=['opinion', 'policy', 'stance', 'values_string',
                                                                    'example_string'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "opinion": opinion,
            "policy": policy,
            "stance": stance,
            "values_string": values_string,
            "example_string": example_string
        })

        return answer

    @staticmethod
    def prompt_for_justifications(policy, opinion, stance, llm, l1_values):
        template = (
            "Q: Someone is {stance} the idea {policy}, arguing that {opinion}. The human values demonstrated via their opinion are:\n"
        )
        for value in l1_values:
            template += "Value: " + value["name"] + "\n"

        template += "Your task is to formulate a justification for why each of the values applies to the argument.\n### ANSWER FORMAT GUIDE ###\nValue: <the value>\nJustification: <A justification of maximum two sentences>"

        prompt = PromptTemplate(template=template, input_variables=['opinion', 'policy', 'stance'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "opinion": opinion,
            "policy": policy,
            "stance": stance,
        })

        return answer

    @staticmethod
    def prompt_for_justifications_new(examples_string, llm):
        template = (
            "{examples_string}"
            "For each of the examples above, your task is to determine the line of reasoning used to arrive at the values, filling the empty Reasoning: entries."
            "### ANSWER FORMAT ###"
            "Q: <the posed argument>"
            "Value: <the value>"
            "Reasoning: <maximum 40 words>"
        )

        prompt = PromptTemplate(template=template, input_variables=['examples_string'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "examples_string": examples_string
        })

        return answer

    @staticmethod
    def few_shot_cot_prompt(policy, opinion, stance, llm, values, example_string):
        template = (
            "### VALUE LIST ###\n"
            "{values_string}\n"
            "{example_string}"
            "Q: Someone is {stance} the idea {policy}, arguing that {opinion}"
        )

        values_string = Processing.stringify_values_for_prompt(values, "Level 1")

        prompt = PromptTemplate(template=template, input_variables=['opinion', 'policy', 'stance', 'values_string',
                                                                    'example_string'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "opinion": opinion,
            "policy": policy,
            "stance": stance,
            "values_string": values_string,
            "example_string": example_string
        })

        return answer

    @staticmethod
    def few_shot_cot_sequence(test_arguments, train_arguments, llm, values, labels_level1, num_examples, verbose):
        test_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                [test_arguments, labels_level1])

        train_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                 [train_arguments, labels_level1])

        values_dict = Processing.get_values_dict(values)
        level_1_values = set(value['name'] for value in values_dict["Level 1"])

        predicted_labels = test_arguments.copy()
        combined_list = []
        for s in [level_1_values]:
            combined_list.extend(s)
        predicted_labels[combined_list] = 0

        for index, row in test_arguments.iterrows():
            policy = row['Conclusion']
            stance = row['Stance']
            opinion = row['Premise']

            if verbose is True:
                print(f"index: {index}, policy: {policy}, opinion: {opinion}, stance: {stance}", "\n")

            example_arguments = Processing.select_examples_fewshot(train_arguments, values_dict, "Level 1", num_examples)
            examples_list = []

            for example_index, example_row in example_arguments.iterrows():

                example_policy = example_row['Conclusion']
                example_stance = example_row['Stance']
                example_opinion = example_row['Premise']
                level_1_values = [value for value in [level_1_values] if example_arguments.at[example_index, value] == 1]

                example_overview = {'policy': example_policy, 'opinion': example_opinion, 'stance': example_stance, 'l1_values': level_1_values}
                examples_list.append(example_overview)

            # Step 1: Prompt with all the examples to gather justifications
            example_string = Processing.individual_example_prompt_string_cot(examples_list)
            llm_justifications = Prompting.prompt_for_justifications_new(example_string, llm)

            # With the example_string of all examples, justifications (their chain-of-thought) and their values, prompt for the unseen test set datapoint
            llm_answer = Prompting.few_shot_cot_prompt(policy, opinion, stance, llm, values, llm_justifications)
            predicted_values = Processing.parse_values_generated(llm_answer)

            predicted_labels.iloc[index, predicted_labels.columns.isin(predicted_values)] = 1

        combined_list.append("Part")
        predicted_labels = predicted_labels[combined_list]
        true_labels = test_arguments[combined_list]

        return predicted_labels, true_labels

    # Performs the few-shot prompting sequence for all the arguments given in as input.
    # @param test_arguments The arguments to be input to the LLM (test set)
    # @param train_arguments The train set arguments to be sampled from as examples
    # @param values The values to be considered
    # @param llm The Large Language Model used
    # @param labels_level1 The true labels for level 1 values
    # @param labels_level2 The true labels for level 2 values
    # @param labels_level3 The true labels for level 3 values
    # @param labels_level4a The true labels for level 4a values
    # @param labels_level4b The true labels for level 4b values
    # @param num_examples The number of examples to be provided in each prompt
    # @param verbose Boolean to indicate if print statements are needed
    # @return Predicted Labels and the arguments with the true labels
    @staticmethod
    def few_shot_sequence(test_arguments, train_arguments, values, llm, labels_level1, labels_level2, labels_level3,
                          labels_level4a, labels_level4b, num_examples, verbose):
        test_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                [test_arguments, labels_level1, labels_level2, labels_level3, labels_level4a, labels_level4b])

        train_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                 [train_arguments, labels_level1, labels_level2, labels_level3, labels_level4a, labels_level4b])

        values_dict = Processing.get_values_dict(values)

        level_1_values = set(value['name'] for value in values_dict["Level 1"])
        level_2_values = values_dict["Level 2"]
        level_3_values = values_dict["Level 3"]
        level_4a_values = values_dict["Level 4A"]
        level_4b_values = values_dict["Level 4B"]

        predicted_labels = test_arguments.copy()
        combined_list = []
        for s in [level_1_values, level_2_values, level_3_values, level_4a_values, level_4b_values]:
            combined_list.extend(s)
        predicted_labels[combined_list] = 0

        for index, row in test_arguments.iterrows():
            policy = row['Conclusion']
            stance = row['Stance']
            opinion = row['Premise']

            all_predicted_values = []

            if verbose is True:
                print(f"index: {index}, policy: {policy}, opinion: {opinion}, stance: {stance}", "\n")

            for level in ["Level 1", "Level 2", "Level 3", "Level 4A", "Level 4B"]:
                example_train_args = Processing.select_examples_fewshot(train_arguments, values_dict, level, num_examples)
                example_string = Processing.generate_example_string_fewshot(example_train_args, values_dict, level)

                llm_result = Prompting.few_shot_prompt(policy, opinion, stance, llm, values, level, example_string)
                predicted_values = llm_result.strip().split('; ')

                all_predicted_values.extend(predicted_values)

            print(all_predicted_values)
            predicted_labels.iloc[index, predicted_labels.columns.isin(all_predicted_values)] = 1

        combined_list.append("Part")
        predicted_labels = predicted_labels[combined_list]
        true_labels = test_arguments[combined_list]

        return predicted_labels, true_labels


    # Performs the zero-shot prompting sequence for all the arguments given in as input.
    # @param arguments The arguments to be input to the LLM
    # @param values The moral values to be considered
    # @param llm The Large Language Model used
    # @param value_labels the true labels
    # @param verbose Boolean to indicate if print statements are needed
    # @return Predicted Labels and the arguments with the true labels
    @staticmethod
    def zero_shot_sequence(arguments, values, llm, labels_level1, labels_level2, labels_level3, labels_level4a,
                           labels_level4b, verbose):

        arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                           [arguments, labels_level1, labels_level2, labels_level3, labels_level4a, labels_level4b])

        values_dict = Processing.get_values_dict(values)
        level_1_values = set(value['name'] for value in values_dict["Level 1"])
        level_2_values = values_dict["Level 2"]
        level_3_values = values_dict["Level 3"]
        level_4a_values = values_dict["Level 4A"]
        level_4b_values = values_dict["Level 4B"]

        predicted_labels = arguments.copy()
        combined_list = []
        for s in [level_1_values, level_2_values, level_3_values, level_4a_values, level_4b_values]:
            combined_list.extend(s)
        predicted_labels[combined_list] = 0

        for index in range(len(arguments)):
            policy = arguments.at[index, 'Conclusion']
            stance = arguments.at[index, 'Stance']
            opinion = arguments.at[index, 'Premise']

            all_predicted_values = []

            if verbose is True:
                print(f"index: {index}, policy: {policy}, opinion: {opinion}, stance: {stance}")

            for level in ["Level 1", "Level 2", "Level 3", "Level 4A", "Level 4B"]:
                llm_result = Prompting.zero_shot_prompt(policy, opinion, stance, llm, values, level)
                predicted_values = llm_result.strip().split('; ')
                all_predicted_values.extend(predicted_values)

            print(all_predicted_values)
            predicted_labels.iloc[index, predicted_labels.columns.isin(all_predicted_values)] = 1

        predicted_labels = predicted_labels[combined_list]
        true_labels = arguments[combined_list]

        return predicted_labels, true_labels
