import torch
from transformers import (MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
                          AutoModelForSequenceClassification, AutoTokenizer)

from .model import Model


class ClassificationModel(Model):
    """
    Class for the classification model
    """

    def __init__(
        self, name, model_name, tokenizer_name, device, quantization, onnx, onnx_convert_kwargs,
    ):
        super().__init__(
            name,
            AutoModelForSequenceClassification,
            model_name,
            tokenizer_name,
            device,
            quantization,
            onnx,
            onnx_convert_kwargs,
            "sentiment-analysis",
        )
        self.check_model_type(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING)

    def _predict(self, x):
        x = x[0]
        pt_batch = self.tokenizer(
            x,
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(**self.prepare_inputs(pt_batch))
        return outputs[0].cpu().numpy().argmax(axis=-1).tolist()
