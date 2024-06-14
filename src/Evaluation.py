from sklearn.metrics import precision_recall_fscore_support


class Evaluation:

    def __init__(self):
        pass

    @staticmethod
    def evaluate(predicted_labels, true_labels, values):
        overview = {}
        overview["overall"] = Evaluation.evaluate_part(predicted_labels.copy(), true_labels.copy(), values)

        usa_pred = predicted_labels[predicted_labels["Part"] == "usa"]
        usa_true = true_labels[predicted_labels["Part"] == "usa"]
        overview["usa"] = Evaluation.evaluate_part(usa_pred, usa_true, values)

        africa_pred = predicted_labels[predicted_labels["Part"] == "africa"]
        africa_true = true_labels[predicted_labels["Part"] == "africa"]
        overview["africa"] = Evaluation.evaluate_part(africa_pred, africa_true, values)

        china_pred = predicted_labels[predicted_labels["Part"] == "china"]
        china_true = true_labels[predicted_labels["Part"] == "china"]
        overview["china"] = Evaluation.evaluate_part(china_pred, china_true, values)

        india_pred = predicted_labels[predicted_labels["Part"] == "india"]
        india_true = true_labels[predicted_labels["Part"] == "india"]
        overview["india"] = Evaluation.evaluate_part(india_pred, india_true, values)

        return overview

    # Prepares an overview of precision, recall and F1 score for a list of predictions made by the LLM, against the true
    # value classifications
    # Also provides the aforementioned statistics by geographical group (USA, Africa, China and India)
    # @param predicted_labels The predictions
    # @param true_labels The true values for the labels
    # @param value_name_list The value names
    # @return A dictionary with the values of different measures
    @staticmethod
    def evaluate_part(predicted_labels, true_labels, values):
        if len(predicted_labels) == 0:
            return
        levels = ["Level 1", "Level 2", "Level 3", "Level 4A", "Level 4B"]
        overview = {}
        zero_cols = true_labels.columns[true_labels.eq(0).all()]
        true_labels_eval = true_labels.drop(zero_cols, axis=1)
        predicted_labels_eval = predicted_labels.drop(zero_cols, axis=1)

        for level in levels:
            if level == "Level 1":
                value_list = [value['name'] for value in values[level] if value['name'] in true_labels_eval]
            else:
                value_list = [value for value in values[level] if value in true_labels_eval]


            l_pred = predicted_labels_eval[value_list].copy()
            l_true = true_labels_eval[value_list].copy()
            overview[level] = {
                "Macro-averaged": precision_recall_fscore_support(l_true, l_pred, average="macro"),
                "Per-value": dict(
                    zip(value_list, precision_recall_fscore_support(l_true, l_pred, average=None)[2]))}

        return overview
