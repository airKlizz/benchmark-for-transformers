import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model import Model


class ClassificationModel(Model):
    """
    Class for the classification model
    """

    def __init__(
        self, name, model_name, tokenizer_name, quantization, device,
    ):
        super().__init__(name, AutoModelForSequenceClassification, model_name, tokenizer_name, device, quantization)

    def _predict(self, x):
        x = x[0]
        pt_batch = self.tokenizer(
            x, padding="longest", truncation=True, max_length=self.tokenizer.max_len, return_tensors="pt",
        )
        with torch.no_grad():
            outputs = self.model(
                input_ids=pt_batch["input_ids"].to(self.device),
                attention_mask=pt_batch["attention_mask"].to(self.device),
            )
        return outputs[0].numpy().argmax(axis=-1).tolist()
