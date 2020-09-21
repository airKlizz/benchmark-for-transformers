try:
    from transformers import BartForSequenceOrdering
except:
    print(
        "Warning: If you try to use BartForSequenceOrdering, you are using the wring python env. Please make sure to use the one with the branch bart-for-token-classification"
    )

from ..model import Model


class OrderingWithFixNumSequencesModel(Model):
    """
    Class for BART for the ordering model
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
        ordering_parameters={},
        num_sequences=None,
    ):
        super().__init__(
            name, BartForSequenceOrdering, model_name, tokenizer_name, device, quantization, onnx, onnx_convert_kwargs
        )
        self.ordering_parameters = ordering_parameters
        self.num_sequences = num_sequences

    def _predict(self, x):

        y = x[1]
        x = x[0]

        x = self.reduce_num_sequences(x, y)

        pt_batch = self.tokenizer(
            [" </s> <s> ".join(sequences) + " </s> <s>" for sequences in x],
            padding=True,
            truncation=True,
            max_length=self.tokenizer.max_len,
            return_tensors="pt",
        )
        outputs = self.model.order(
            input_ids=pt_batch["input_ids"].to(self.device),
            attention_mask=pt_batch["attention_mask"].to(self.device),
            **self.ordering_parameters,
        )
        for output, sequences in zip(outputs, x):
            output.remove(max(output))
            for i in range(len(sequences)):
                if i not in output:
                    output.append(i)
            while max(output) > len(sequences) - 1:
                print(
                    f"INFO: Before second verification: sequences: {len(sequences)} - output: {len(output)} --- \n output:\n{output}"
                )
                output.remove(max(output))
            assert len(output) == len(sequences), f"sequences: {sequences} - output: {output}"
        return outputs

    def _prepare_reference(self, references):
        if self.num_sequences == None:
            return references
        references = references[: self.num_sequences]
        updated_labels = dict(zip(sorted(references), range(self.num_sequences)))
        references = [updated_labels[label] for label in references]
        return references

    def reduce_num_sequences(self, x, y):
        if self.num_sequences == None:
            return x
        for _x, _y in zip(x, y):
            idx_to_remove = _y[self.num_sequences :]
            idx_to_remove.sort(reverse=True)
            for idx in idx_to_remove:
                _x.pop(idx)
        return x
