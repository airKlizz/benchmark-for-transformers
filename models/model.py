import time

import torch.quantization
from tqdm import tqdm
from transformers import AutoTokenizer


class Model(object):
    """
    Parent class of all sub-models
    """

    def __init__(self, name, model_cls, model_name, tokenizer_name, device, quantization):
        self.name = name
        self.device = device
        if tokenizer_name == None:
            tokenizer_name = model_name
        self.model = model_cls.from_pretrained(model_name).eval().to(self.device)
        if quantization == "dynamic":
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def predict(self, dataset, x_column_name, batch_size):
        predictions = []
        inference_time_measures = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Prediction"):
            x = [dataset[i : i + batch_size][column_name] for column_name in x_column_name]
            start = time.time()
            predictions += self._predict(x)
            time_elapsed = (time.time() - start) / batch_size
            inference_time_measures.append(time_elapsed)
        return predictions, inference_time_measures

    def _predict(self, x):
        raise NotImplementedError

    def prepare_references(self, references):
        return list(map(self._prepare_reference, references))

    def _prepare_reference(self, reference):
        return reference
