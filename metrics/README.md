# Metrics

We use the [``nlp``](https://github.com/huggingface/nlp) library from Huggingface to compute evaluation.

This library provides many metrics that you can find [here](https://huggingface.co/metrics).

## Example of usage

### Classification

```python
from nlp import load_metric

metric = load_metric("glue", "sst2")

references = [1, 4, 2]
predictions = [1, 3, 2]

score = metric.compute(predictions, references)
print(score)
# {'accuracy': 0.6666666666666666}
```

### Summarization

```python
from nlp import load_metric

metric = load_metric("rouge")

references = ["The weither is nice today."]
predictions = ["The weither is bad today."]

score = metric.compute(predictions, references, rouge_types=["rouge1", "rouge2", "rougeL"])
print(score)
# {'rouge1': AggregateScore(low=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002), mid=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002), high=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002)), 'rouge2': AggregateScore(low=Score(precision=0.5, recall=0.5, fmeasure=0.5), mid=Score(precision=0.5, recall=0.5, fmeasure=0.5), high=Score(precision=0.5, recall=0.5, fmeasure=0.5)), 'rougeL': AggregateScore(low=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002), mid=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002), high=Score(precision=0.8, recall=0.8, fmeasure=0.8000000000000002))}
```

### NER

```python
from nlp import load_metric

metric = load_metric("seqeval")

references = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
predictions = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]

score = metric.compute(predictions, references)
print(score)
# {'PER': {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'number': 1}, 'MISC': {'precision': 0.0, 'recall': 0.0, 'f1': 0, 'number': 1}, 'overall_precision': 0.5, 'overall_recall': 0.5, 'overall_f1': 0.5, 'overall_accuracy': 0.8}
```

## Add a metric

If you want to use a custom metric, you have to create a metric that can be load using ``nlp.load_metric``. 
