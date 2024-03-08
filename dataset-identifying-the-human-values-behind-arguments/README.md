# Identifying the Human Values behind Arguments

Anonymized as currently under blind review.

The dataset contains the following:

## Value Taxonomy
Our proposed [value taxonomy](values.json) of 54 values in JSON format:
```
{
  "values": [
    {
      "alternativeName": "<the original name of each value from the literature>",
      "name": "<the name of each value as used in the crowdsourcing study>,
      "id": "<a unique 4-character identifier for the value>",
      "level2": "<value category this value belongs to>",
      "level3": [ "Openness to change", "Self-enhancement", "Conservation", "Self-transcendence" ], # one or two of these; to which this value belongs to
      "level4a": [ "Social focus", "Personal focus" ], # one of these; to which this value belongs to
      "level4b": [ "Growth, Anxiety-free", "Self-protection, Anxiety avoidance" ], # one or both of these; to which this value belongs to
      "descriptions": [
        "<example effect that a matching arguments might imply>",
        ...
      ],
      "related": [ # optional
        {
          "id": "<identifier of the related value>",
          "descriptions": [
            "<example effect that one might associate with this value, but rather indicates the related value>",
            ...
          ]
        }, ...
      ]
    }, ...
  ]
}
```


## Argument Corpus

### Annotation Results
The annotated corpus of 5270 arguments in tab separated value format:
- [`arguments.tsv`](arguments.tsv): Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - `Part`: Name of the containing dataset part from the paper; one of "africa", "china", "india", "usa"
    - `Usage`: Name of the set the argument is used for in the machine learning experiments; one of "train", "validation" or "test"
    - `Conclusion`: Conclusion text of the argument
    - `Stance`: Stance of the `Premise` towards the `Conclusion; one of "in favor of", "against"
    - `Premise`: Premise text of the argument
- [`labels-level1.tsv`](labels-level1.tsv) / [`labels-level2.tsv`](labels-level2.tsv) / [`labels-level3.tsv`](labels-level3.tsv) / [`labels-level4a.tsv`](labels-level4a.tsv) / [`labels-level4b.tsv`](labels-level4b.tsv): Each row corresponds to one argument
    - `Argument ID`: The unique identifier for the argument
    - Other: The column name specifies a label in that level, and the value whether the argument has that label (1) or not (0)

### Origin Data
Extra information regarding the origin of each dataset part:
- [`origin-africa.tsv`](origin-africa.tsv): The 50 arguments extracted from editorials of the Debating Ideas section at [https://africanarguments.org](https://africanarguments.org)
    - `Argument ID`: The unique identifier for the argument
    - `URL`: Link to the editorial the argument was extracted from
- [`origin-china.tsv`](origin-china.tsv): The 100 arguments from the Chinese question-answering website [Zhihu](https://www.zhihu.com)
    - `Argument ID`: The unique identifier for the argument
    - `Conclusion (chinese)`: The original chinese conclusion statement
    - `Premise (chinese)`: The original chinese premise statement
    - `URL`: Link to the original statement
- [`origin-india.tsv`](origin-india.tsv): The 100 arguments from Controversial Debate Topics 2021 section at [https://www.groupdiscussionideas.com](https://www.groupdiscussionideas.com)
    - `Argument ID`: The unique identifier for the argument
    - `URL`: Link to the topic an argument was taken from
- [`origin-usa.tsv`](origin-usa.tsv): The 5020 arguments taken from the IBM corpus [IBM-ArgQ-Rank-30kArgs](https://research.ibm.com/haifa/dept/vst/debating_data.shtml#Argument%20Quality)
    - `Argument ID`: The unique identifier for the argument
    - `WA`: the quality label according to the weighted-average scoring function
    - `MACE-P`: the quality label according to the MACE-P scoring function
    - `stance_WA`: the stance label according to the weighted-average scoring function
    - `stance_WA_conf`: the confidence in the stance label according to the weighted-average scoring function

## Machine Learning Results
The prediction scores for each method in tab separated value format:
- [`evaluation.tsv`](evaluation.tsv): Each row corresponds to one result; rows only exist if the `Label` was assigned at least once for the `Test dataset`
    - `Method`: Name of the method that produced the result; one of "BERT", "SVM", "1-Baseline"
    - `Test dataset`: Name of the dataset part on which the result was produced; one of "africa", "china", "india", "usa"
    - `Level`: Identifier of the label's level; one of "1", "2", "3", "4a", "4b"
    - `Label`: Name of the label for which the result was produced; "Mean" for the line indicating the averaged scores on the combination of `Method`, `Test dataset` and `Level`
    - `Precision`: Precision that the `Method` reached for identifying the `Label` on the `Test dataset`
    - `Recall`: Recall that the `Method` reached for identifying the `Label` on the `Test dataset`
    - `F1`: F1-score that the `Method` reached for identifying the `Label` on the `Test dataset`
    - `Accuracy `: Accuracy that the `Method` reached for identifying the `Label` on the `Test dataset`


## Crowdsourcing Interface
The `crowdsourcing` folder contains:
- [mturk-worker-template-example](crowdsourcing/mturk-worker-template-example.html): The interface with five example arguments as a showcase
- [mturk-worker-template.html](crowdsourcing/mturk-worker-template.html): The interface template used in our crowdsourcing studies on [Amazon Mechanical Turk](https:://www.mturk.com).


## Analysis
The ``analysis` folder contains scripts to reproduce analyses from the paper:
- [`list-value-frequencies.R`](analysis/list-value-frequencies.R): Produces the frequencies used for Table 1
- [`plot-density-number-of-labels.R`](analysis/plot-density-number-of-labels.R): Produces Figure 2
- [`plot-parallel-coordinates.R`](analysis/plot-parallel-coordinates.R): Produces Figure 4
- [`significance-tests.R`](analysis/significance-tests.R): Produces the significance analysis for Section 5

