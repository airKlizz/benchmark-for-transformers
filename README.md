# Benchmark for Transformers

Evaluate performance of Transformers in different scenarios. The library is mainly based on the work of the [🤗](https://huggingface.co/) team and should be used if you already use their libraries.

## Installation

Install using pip:

```bash
pip install benchmark-for-transformers
```

## Quick tour

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1CnbslD5QEkmP-_fqtBw-HwzBjupJRLb7?usp=sharing)

``benchmark-for-transformers`` allows you to create a benchmark to evaluate and compare Transformers in a scenario.

You have to create a benchmark that is composed of:

- one dataset
- one or more metrics
- one or more scenarios

To create a benchmark you can either use the API or a ``json`` file.

### Create a benchmark using the API

```python
from benchmark_for_transformers import Benchmark
import torch

torch.set_num_threads(1)

# Set the dataset and the metric to use for the Benchmark
benchmark = Benchmark.from_args(
    dataset_name="xsum",
    dataset_split="test[:10]",
    x_column_name=["document"],
    y_column_name="summary",
    metric_name="rouge",
    metric_values=["rouge1", "rouge2", "rougeL"],
    metric_run_kwargs={"rouge_types": ["rouge1", "rouge2", "rougeL"]},
)

# Add a scenario
benchmark.reset_scenarios()
benchmark.add_scenario(
    name="Bart Xsum on cuda",
    model_class="summarization",
    model_name="facebook/bart-large-xsum",
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
    device="cuda"
)

df = benchmark.run()
print(df)
#  	# of parameters 	latency (mean) 	latency (90th percentile) 	rouge_rouge1 	rouge_rouge2 	rouge_rougeL
# Bart Xsum on cuda 	406290432 	0.850256 	0.941304 	0.376018 	0.118984 	0.274553
```

### Create a benchmark using json file

The benchmark ``json`` file takes the same arguments as the API.

For example, ``sst-2.json`` is a benchmark file for the Sentiment Analysis dataset:

```json
{
    "scenarios": [
        {
            "name": "distilbert",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 1,
            "device": "cuda"
        },
        {
            "name": "albert-base",
            "model_class": "classification",
            "model_name": "textattack/albert-base-v2-SST-2",
            "tokenizer_name": "textattack/albert-base-v2-SST-2",
            "batch_size": 1,
            "device": "cuda"
        },
        {
            "name": "bert base",
            "model_class": "classification",
            "model_name": "textattack/bert-base-uncased-SST-2",
            "batch_size": 1,
            "device": "cuda"
        }
    ],
    "dataset": {
        "dataset_name": "glue",
        "split": "validation",
        "x_column_name": ["sentence"],
        "y_column_name": "label",
        "init_kwargs": {"name": "sst2"}
    },
    "metrics": [
        {
            "metric_name": "glue",
            "values": ["accuracy"],
            "init_kwargs": {"config_name": "sst2"}
        }
    ]
}
```

Once the benchmark file is ready, you can either load it using the API or directly run it using the CLI.

#### Run the ``json`` file using API

```python
from benchmark_for_transformers import Benchmark

benchmark = Benchmark.from_json("sst-2.json")

df = benchmark.run()
print(df)
#  	# of parameters 	latency (mean) 	latency (90th percentile) 	glue_accuracy
# distilbert 	66955010 	0.006111 	0.007480 	0.910550
# albert-base 	11685122 	0.012642 	0.014657 	0.925459
# bert base 	109483778 	0.010371 	0.012245 	0.924312
```

#### Run the ``json`` file using CLI

```bash
benchmark-for-transformers-run --run_args_file "sst-2.json" --verbose --csv_file "results.csv"
#  	# of parameters 	latency (mean) 	latency (90th percentile) 	glue_accuracy
# distilbert 	66955010 	0.006111 	0.007480 	0.910550
# albert-base 	11685122 	0.012642 	0.014657 	0.925459
# bert base 	109483778 	0.010371 	0.012245 	0.924312
```

## Supported features

### Datasets and metrics

``benchmark-for-transformers`` uses the [``datasets``](https://github.com/huggingface/datasets/) to load datasets and metrics. Therefore you can use all the datasets and metrics avalaible in this library. If you want to use a dataset or a metric that is not include in ``datasets``, you can easily add it by creating a small script (see documentation to add a [dataset](https://huggingface.co/docs/datasets/add_dataset.html) or a [metric](https://huggingface.co/docs/datasets/add_metric.html)). For more information see the ``datasets`` [documentation](https://huggingface.co/docs/datasets/).

### Tasks

For the moment, ``benchmark-for-transformers`` only supports 4 tasks:

- [Classification](benchmark_for_transformers/models/classification.py),
- [NER](benchmark_for_transformers/models/ner.py),
- [Summarization](benchmark_for_transformers/models/summarization.py),
- [Ordering](benchmark_for_transformers/models/ordering.py) (this can not be used for the moment and is for internal use).

These class are based on the main [Model](benchmark_for_transformers/model.py) class and use HuggingFace [``transformers``](https://github.com/huggingface/transformers/) models. 

You can add a new task by creating a task script and put the path to this script in the ``model_class`` ``Scenario`` argument.

### Optimization

You can define several optimization features in the scenario:

- batch size,
- quantization,
- ONNX support.

You can also define the device you want to use.

For example, let's try some optimization features on ``distilbert`` on the Sentiment Analysis dataset.

First we define a new benchmark ``json`` file:

```json
{
    "scenarios": [
        {
            "name": "distilbert on cpu",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 1,
            "device": "cpu"
        },
        {
            "name": "distilbert on cuda",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 1,
            "device": "cuda"
        },
        {
            "name": "distilbert on cpu bsz 8",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 8,
            "device": "cpu"
        },
        {
            "name": "distilbert on onnx cpu bsz 8",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 8,
            "device": "cpu",
            "onnx": true
        },
        {
            "name": "quantized distilbert on onnx cpu bsz 8",
            "model_class": "classification",
            "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
            "tokenizer_name": "distilbert-base-uncased",
            "batch_size": 8,
            "device": "cpu",
            "onnx": true,
            "quantization": true
        }
    ],
    "dataset": {
        "dataset_name": "glue",
        "split": "validation",
        "x_column_name": ["sentence"],
        "y_column_name": "label",
        "init_kwargs": {"name": "sst2"}
    },
    "metrics": [
        {
            "metric_name": "glue",
            "values": ["accuracy"],
            "init_kwargs": {"config_name": "sst2"}
        }
    ]
}
```

Then, we run it using the API:

```python
from benchmark_for_transformers import Benchmark

benchmark = Benchmark.from_json("sst-2-optimization.json")

df = benchmark.run()
print(df)
#  	# of parameters 	latency (mean) 	latency (90th percentile) 	glue_accuracy
# distilbert on cpu 	            66955010 	0.061905 	0.074103 	0.910550
# distilbert on cuda 	            66955010 	0.005782 	0.006732 	0.910550
# distilbert on cpu bsz 8 	        66955010 	0.035685 	0.043952 	0.910550
# distilbert on onnx cpu bsz 8 	            -1 	0.036746 	0.044342 	0.910550
# quantized distilbert on onnx cpu bsz 8 	-1 	0.023608 	0.029647 	0.902523
```

## Examples

Some examples benchmark ``json`` files are in the [examples](/examples) folder. You can look at it to see how use ``benchmark-for-transformers``.

In the [examples](/examples) folder, there are also subfolders containing examples of personnalized datasets and metrics scripts.

## Documentation

You can find a description of the repository, guide and examples in the [documentation](https://remi-calizzano.gitbook.io/benchmark-for-transformers/).