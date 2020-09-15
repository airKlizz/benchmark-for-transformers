import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .model import Model


class NerModel(Model):
    """
    Class for the classification model
    """

    equivalent_labels = {
        "B-location": "B-LOC",
        "I-location": "I-LOC",
        "B-person": "B-PER",
        "I-person": "I-PER",
    }

    def __init__(
        self, name, model_name, tokenizer_name, device, quantization, equivalent_labels=equivalent_labels,
    ):
        super().__init__(name, AutoModelForTokenClassification, model_name, tokenizer_name, device, quantization)
        self.equivalent_labels = equivalent_labels

    def _predict(self, x):
        all_tokens = x[0]
        pt_batch = self.tokenizer(
            [" ".join(tokens) for tokens in all_tokens],
            padding="longest",
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=pt_batch["input_ids"].to(self.device),
                attention_mask=pt_batch["attention_mask"].to(self.device),
            )
        outputs = outputs[0].cpu().numpy().argmax(axis=-1).tolist()
        outputs = [output[1:-1] for output in outputs]  # remove first and last tokens
        assert len(outputs) == len(all_tokens)
        return [self._get_predictions(all_tokens[i], outputs[i]) for i in range(len(outputs))]

    def _get_predictions(self, tokens, output):
        offsets = [len(self.tokenizer.tokenize(tok)) for tok in tokens]
        assert sum(offsets) == len(
            output
        ), f"Number of input tokens ({sum(offsets)}) and number of output tokens ({len(output)}) are different"
        predictions = [output[sum(offsets[:i]) : sum(offsets[:i]) + offset] for i, offset in enumerate(offsets)]
        predictions = [
            self.model.config.id2label[max(set(prediction), key=prediction.count)] if prediction != [] else "O"
            for prediction in predictions
        ]
        return [
            self.equivalent_labels[pred] if pred in self.equivalent_labels.keys() else pred for pred in predictions
        ]

    def _prepare_reference(self, reference):
        return [self.equivalent_labels[ref] if ref in self.equivalent_labels.keys() else ref for ref in reference]
