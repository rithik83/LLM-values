import math
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


class Evaluation:

    def __init__(self):
        pass

    @staticmethod
    def average_pairwise_jaccard(df):
        combinations = math.comb(len(df), 2)
        avg_jaccard = 0

        for i in range(len(df) - 1):
            for j in range(i + 1, len(df)):
                avg_jaccard += jaccard_score(df.iloc[i], df.iloc[j], average='macro')

        return float(avg_jaccard) / float(combinations)

    # Prepares an overview of F1 score metrics for LLM predictions
    # @param predicted_labels The predictions
    # @param true_labels The true values for the labels
    # @param value_name_list The value names
    # @return A dictionary with the values of different f1 scores
    @staticmethod
    def f1_overview(predicted_labels, true_labels, value_name_list):
        overview = {
            "Micro-averaged F1 score": f1_score(true_labels, predicted_labels, average="micro", zero_division=1.0),
            "Macro-averaged F1 score": f1_score(true_labels, predicted_labels, average="macro", zero_division=1.0),
            "Weighted average F1 score": f1_score(true_labels, predicted_labels, average="weighted", zero_division=1.0),
            "Per-value F1 score": dict(
                zip(value_name_list, f1_score(true_labels, predicted_labels, average=None, zero_division=1.0)))}

        return overview
