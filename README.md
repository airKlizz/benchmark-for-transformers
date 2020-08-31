# Transformers on CPU

Evaluate performance of Transformers in different scenarios.

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnbslD5QEkmP-_fqtBw-HwzBjupJRLb7?usp=sharing)

### Load parameters directly using the api

To initialize the benchmark, run:

```python
from benchmark.benchmark import Benchmark

torch.set_num_threads(1)

benchmark = Benchmark.from_args(
    dataset_name="xsum",
    dataset_split="test[:10]",
    x_column_name=["document"],
    y_column_name="summary",
    metric_name="rouge",
    metric_values=["rouge1", "rouge2", "rougeL"],
    metric_run_kwargs={"rouge_types": ["rouge1", "rouge2", "rougeL"]},
)

benchmark.add_scenario(
    name="Scenario1",
    model_class="summarization",
    model_name="sshleifer/distilbart-xsum-1-1",
    tokenizer_name="facebook/bart-large",
    init_kwargs={
        "generation_parameters": {
            "num_beams": 4,
            "length_penalty": 0.5,
            "min_length": 11,
            "max_length": 62
        }
    },
    batch_size=1,
    quantization="dynamic"
)

df = benchmark.run()
print(df)
#           mean    90e centile rouge_rouge1    rouge_rouge2    rouge_rougeL
# Scenario1 2.436136    2.7086  0.176576    0.005405    0.141547
```

### Load parameters using ``json`` file

Example of ``json`` file:

```json
{
    "scenarios": [
        {
            "name": "Scenario1",
            "model_class": "summarization",
            "model_name": "sshleifer/distilbart-xsum-1-1",
            "tokenizer_name": "facebook/bart-large",
            "init_kwargs": {
                "generation_parameters": {
                    "num_beams": 4,
                    "length_penalty": 0.5,
                    "min_length": 11,
                    "max_length": 62
                }
            },
            "batch_size": 1
        },
        {
            "name": "Scenario2",
            "model_class": "summarization",
            "model_name": "sshleifer/distilbart-xsum-1-1",
            "tokenizer_name": "facebook/bart-large",
            "init_kwargs": {
                "generation_parameters": {
                    "num_beams": 4,
                    "length_penalty": 0.5,
                    "min_length": 11,
                    "max_length": 62
                }
            },
            "batch_size": 1,
            "quantization": "dynamic"
        }
    ],
    "dataset": {
        "dataset_name": "xsum",
        "split": "test[:10]",
        "x_column_name": ["document"],
        "y_column_name": "summary"
    },
    "metric": {
        "metric_name": "rouge",
        "values": ["rouge1", "rouge2", "rougeL"],
        "run_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL"]}
    }
}
```

Initialize the benchmark using ``from_json``:

```python
from benchmark.benchmark import Benchmark

torch.set_num_threads(1)
benchmark = Benchmark.from_json("path/to/json/file")
df = benchmark.run()
print(df)
#           mean    90e centile rouge_rouge1    rouge_rouge2    rouge_rougeL
# Scenario1 2.436136    2.7086  0.176576    0.005405    0.141547
# Scenario2 2.421537 	2.6760 	0.178549 	0.005405 	0.142942
```

## Features

### Tasks

- Summarization
- NER
- Text classification
- Ordering

### Optimization features

- Batch size
- Quantization
- Set number of pytorch threads

### Speed metrics

- Mean latency
- 90th pourcentile latency
- Throughput

## Add a dataset

All the datasets from the [``nlp``](https://github.com/huggingface/nlp) library can be used.
If you want to use a local dataset, you can create you own script (see [instructions](https://huggingface.co/nlp/add_dataset.html)) as scripts in ``datasets/``.

## Add a metric

As for the datasets, all the metrics from ``nlp`` can be used.
You can add your metric by creating a script in ``metrics/``.
