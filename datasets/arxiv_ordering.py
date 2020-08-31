# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace NLP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""arXiv ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import nlp
import numpy as np

_CITATION = """
@misc{chen2016neural,
    title={Neural Sentence Ordering},
    author={Xinchi Chen and Xipeng Qiu and Xuanjing Huang},
    year={2016},
    eprint={1607.06952},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

_DESCRIPTION = """
Dataset for sentence ordering using text from arXiv."""

_URL = "https://drive.google.com/uc?export=download&id=0B-mnK8kniGAieXZtRmRzX2NSVDg"

_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class ArXivOrdering(nlp.GeneratorBasedBuilder):
    """arXiv ordering dataset."""

    VERSION = nlp.Version("1.0.0")

    def _info(self):
        info = nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    _SENTENCES: nlp.Sequence(nlp.Value("string")),
                    _SHUFFLED_SENTENCES: nlp.Sequence(nlp.Value("string")),
                    _LABEL: nlp.Sequence(nlp.Value("int8")),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/FudanNLP/NeuralSentenceOrdering",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = dl_manager.download_and_extract(_URL)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"path": os.path.join(data_path, "train.txt")},),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"path": os.path.join(data_path, "valid.txt")},),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"path": os.path.join(data_path, "test.txt")},),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            data = f.read()
            examples = data.split("\n\n")
            for i, example in enumerate(examples):
                lines = example.split("\n")
                sentences = lines[2:]
                if len(sentences) == 0:
                    continue
                shuffled_sentences, label = self.shuffle_sentences(sentences)
                yield i, {
                    _SENTENCES: sentences,
                    _SHUFFLED_SENTENCES: shuffled_sentences,
                    _LABEL: label,
                }

    def shuffle_sentences(self, sentences):
        sentences = np.array(sentences)
        permutation = np.random.permutation(len(sentences))
        return sentences[permutation].tolist(), np.argsort(permutation).tolist()
