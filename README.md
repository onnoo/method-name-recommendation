# method-name-recommendation
Naming source code identifiers is an important issue in software engineering.
Meaningful and concise names of program entities play an important role in code comprehension, and they have great effect on reducing software maintenance management costs.
The nature of these source code components' inconsistent or difficult-to-understand names is called **naming smell** and needs to be modified to so-called good names by refactoring them.
Among these source code identifiers, on average, the most complex identifier is known as the method name.
Recent approaches to generating good method names are deep learning-based **abstract text summarization** techniques, which attempt to overcome the limitations of not generating new names, which are disadvantages of existing methods, using the idea of making text short summaries.

In this paper, we derive the applicability of abstract text summarization approaches to method name generation that is consistent with method content.
Our results show that it can be applied to recommend good method names using abstract text summarization techniques.
We expect this to contribute to the development of a method name generation automation tool for developers or to the study of different types of identifiers generation.

----------

## Contents
1. [Dataset](#dataset)
2. [Experiments](#experiments)

## Dataset

We experiments using three datasets used by the [code2seq](https://github.com/tech-srl/code2seq) paper, with cleaner in [DeepName](https://github.com/deepname2021icse/DeepName-2021-ICSE).

These datasets are available in raw format (i.e., .java files) at https://github.com/tech-srl/code2seq/blob/master/README.md#datasets.

## Experiments

1. The base model is [MNire](https://sonvnguyen.github.io/mnire/) (Encoder-Decoder architecture implemented tensorflow)
2. For comparison, we train abstract text summarization SOTA model from https://github.com/huggingface/transformers/tree/master/examples/pytorch/summarization

|    Dataset    |      Model    |   Precision   |     Recall    |    F-score    |  Exact Match  |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|   Java-small  |      MNire    |      53.7     |      49.0     |      49.7     |      26.5     |
|   Java-small  |    T5-small   |      95.7     |      88.3     |      90.4     |      76.0     |
|    Java-med   |      MNire    |      41.9     |      35.8     |      37.1     |      13.8     |
|    Java-med   |    T5-small   |      90.6     |      82.9     |      85.2     |      66.3     |

