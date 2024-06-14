from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pandas as pd
from Processing import Processing
from functools import reduce


class Prompting:

    def __init__(self):
        pass

    # Performs an individual prompt to the specified LLM, gathering its identified values only
    @staticmethod
    def zero_shot_prompt(policy, opinion, stance, llm, values, level):
        template = (
            "You are an expert in debate, argumentation and moral inclinations."
            "Someone is {stance} the idea {policy}, arguing that {opinion}."
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
            "You are an expert in debate, argumentation and moral inclinations."
            "Someone is {stance} the idea {policy}, arguing that {opinion}."
            "Your task is to determine the values that motivate and form the foundation behind the opinion stated."
            "Choose values from the below list, and do not make up your own values.\n"
            "{values_string}\n"
            "Represent your answer as a semicolon separated list of values. Be as selective and precise as possible."
            "Only choose the human values that very clearly are associated with and align with the argument being made."
            "Do not choose irrelevant or barely related values. Be precise and selective."
            "Following are examples of arguments for which value classification has been performed. Thoroughly understand the reasoning process"
            " that led to the classifications being made, and perform value classification on the last entry."
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
    def prompt_for_justifications_new(examples_string, llm):
        template = (
            "{examples_string}"
            "Each of the examples above represents a person's opinion towards a statement. Provide plausible explanations"
            " for why the person's argument represents the values. Fill the empty Reasoning: entries. We want to identify"
            " why the person's argument represents those values."
            "### ANSWER FORMAT ###"
            "Q: <the posed argument>"
            "Reasoning: <reasoning for why the argument made by the person exhibits the specific value>"
            "Value: <the value>"
            "repeat Reasoning: and Value: for as many values provided"
        )

        prompt = PromptTemplate(template=template, input_variables=['examples_string'])

        runnable = prompt | llm | StrOutputParser()

        answer = runnable.invoke({
            "examples_string": examples_string
        })

        return answer

    @staticmethod
    def few_shot_cot_prompt(policy, opinion, stance, llm, values, level, example_string):
        template = (
            "You are an expert in debate, argumentation and moral inclinations."
            "Someone is {stance} the idea {policy}, arguing that {opinion}."
            "Your task is to determine the values that motivate and form the foundation behind the opinion stated."
            "Choose values from the below list, and do not make up your own values.\n"
            "{values_string}\n"
            "Be as selective and precise as possible."
            "Only choose the human values that very clearly are associated with and align with the argument being made."
            "Do not choose irrelevant or barely related values. Be precise and selective."
            "Following are examples of arguments for which value classification has been performed, and the reasoning "
            "behind selecting the values. "
            "Thoroughly understand the reasoning process that led to the classifications being made, and perform value "
            "classification on the last entry."
            "{example_string}"
            "In your answer for the question that follows, strictly follow the Reasoning:\nValue: format that is observed in the examples."
            "Q: Someone is {stance} the idea {policy}, arguing that {opinion}. What are the human values that motivate their opinion?"
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

    # Performs the Chain-of-Thought prompting sequence for all the arguments given in as input.
    @staticmethod
    def few_shot_cot_sequence(test_arguments, train_arguments, llm, values, levels, labels_level1, labels_level2, num_examples, verbose):
        test_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                [test_arguments, labels_level1, labels_level2])

        train_arguments = reduce(lambda left, right: pd.merge(left, right, on='Argument ID', how='inner'),
                                 [train_arguments, labels_level1, labels_level2])

        values_dict = Processing.get_values_dict(values)

        level_1_values = [value['name'] for value in values_dict["Level 1"]]
        level_2_values = values_dict["Level 2"]

        predicted_labels = test_arguments.copy()
        combined_list = []
        for s in [level_1_values, level_2_values]:
            combined_list.extend(s)
        predicted_labels[combined_list] = 0

        for index, row in test_arguments.iterrows():
            policy = row['Conclusion']
            stance = row['Stance']
            opinion = row['Premise']

            if verbose >= 1:
                print(f"index: {index}, policy: {policy}, opinion: {opinion}, stance: {stance}", "\n")

            all_predicted_values = []

            for level in levels:
                example_arguments = Processing.select_examples_fewshot(train_arguments, values_dict, level, num_examples, index)
                examples_list = []

                for example_index, example_row in example_arguments.iterrows():
                    example_policy = example_row['Conclusion']
                    example_stance = example_row['Stance']
                    example_opinion = example_row['Premise']
                    if level == "Level 1":
                        example_values = [value for value in level_1_values if example_arguments.at[example_index, value] == 1]
                    else:
                        example_values = [value for value in level_2_values if example_arguments.at[example_index, value] == 1]

                    example_overview = {'policy': example_policy, 'opinion': example_opinion, 'stance': example_stance, 'values': example_values}
                    examples_list.append(example_overview)

                # Step 1: Prompt with all the examples to gather justifications
                example_string = Processing.individual_example_prompt_string_cot(examples_list)
                llm_justifications = Prompting.prompt_for_justifications_new(example_string, llm)

                # With the example_string of all examples, justifications (their chain-of-thought) and their values, prompt for the unseen test set datapoint
                llm_answer = Prompting.few_shot_cot_prompt(policy, opinion, stance, llm, values, level, llm_justifications)
                print(llm_answer)
                predicted_values = Processing.parse_values_generated(llm_answer)

                for value in predicted_values:
                    all_predicted_values.append(value)

            print(all_predicted_values)
            predicted_labels.iloc[index, predicted_labels.columns.isin(all_predicted_values)] = 1

        combined_list.append("Part")
        predicted_labels = predicted_labels[combined_list]
        true_labels = test_arguments[combined_list]

        return predicted_labels, true_labels

    # Performs the few-shot prompting sequence for all the arguments given in as input.
    @staticmethod
    def few_shot_sequence(test_arguments, train_arguments, values, llm, levels, labels_level1, labels_level2, labels_level3,
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

            for level in levels:
                example_train_args = Processing.select_examples_fewshot(train_arguments, values_dict, level, num_examples, index)
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
    @staticmethod
    def zero_shot_sequence(arguments, values, llm, levels, labels_level1, labels_level2, labels_level3, labels_level4a,
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

            for level in levels:
                llm_result = Prompting.zero_shot_prompt(policy, opinion, stance, llm, values, level)
                if level == "Level 4B":
                    print("result after prompt for level 4b: ", llm_result)
                predicted_values = llm_result.strip().split('; ')
                all_predicted_values.extend(predicted_values)

            print(all_predicted_values)
            predicted_labels.iloc[index, predicted_labels.columns.isin(all_predicted_values)] = 1

        combined_list.append("Part")
        predicted_labels = predicted_labels[combined_list]
        true_labels = arguments[combined_list]

        return predicted_labels, true_labels
