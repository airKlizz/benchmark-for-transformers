from transformers import AutoModelWithLMHead, AutoTokenizer

from ..model import Model


class SummarizationModel(Model):
    """
    Class for the summarization model
    """

    def __init__(
        self,
        name,
        model_name,
        tokenizer_name,
        device,
        quantization,
        onnx,
        onnx_convert_kwargs,
        generation_parameters={},
        prefix="",
    ):
        super().__init__(
            name, AutoModelWithLMHead, model_name, tokenizer_name, device, quantization, onnx, onnx_convert_kwargs
        )
        self.generation_parameters = generation_parameters
        self.prefix = prefix

    def _predict(self, x):
        x = x[0]
        pt_batch = self.tokenizer(
            [self.prefix + elem for elem in x],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            input_ids=pt_batch["input_ids"].to(self.device),
            attention_mask=pt_batch["attention_mask"].to(self.device),
            **self.generation_parameters,
        )
        summaries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return summaries
