from matplotlib import pyplot as plt


class Plotting:

    def __init__(self):
        pass

    @staticmethod
    def plot_f1_per_value(f1_overview_dict):
        x = list(f1_overview_dict["Per-value F1 score"].keys())
        y = list(f1_overview_dict["Per-value F1 score"].values())

        plt.figure(figsize=(20, 5))
        plt.xticks(rotation=45, ha='right')
        plt.plot(x, y, marker='.', markersize=10, markerfacecolor='red', markeredgecolor='red')
        plt.ylim([0, 1])
        plt.show()
