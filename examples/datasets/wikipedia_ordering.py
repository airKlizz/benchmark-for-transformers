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
"""wikipedia ordering dataset."""

from __future__ import absolute_import, division, print_function

import json
import os

import datasets
import numpy as np

from statistics import mean

_CITATION = """
"""

_DESCRIPTION = """
Dataset for sentence ordering using text from wikipedia."""

#_PATH = "datasets/data/enwiki"
_URL = "https://drive.google.com/uc?export=download&id=1ikeIUzxy0FgjpsbNexQQVbqbpZ0HzUDJ"

_TITLE = "title"
_SECTION_TITLE = "section_title"
_SENTENCES = "sentences"
_SHUFFLED_SENTENCES = "shuffled_sentences"
_LABEL = "label"


class WikipediaOrdering(datasets.GeneratorBasedBuilder):
    """wikipedia ordering dataset."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        info = datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    _TITLE: datasets.Value("string"),
                    _SECTION_TITLE: datasets.Value("string"),
                    _SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _SHUFFLED_SENTENCES: datasets.Sequence(datasets.Value("string")),
                    _LABEL: datasets.Sequence(datasets.Value("int8")),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )
        return info

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = dl_manager.download_and_extract(_URL)
        #data_path = _PATH

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"path": os.path.join(data_path, "enwiki/train.json")},),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"path": os.path.join(data_path, "enwiki/valid.json")},),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"path": os.path.join(data_path, "enwiki/test.json")},),
        ]

    def _generate_examples(self, path=None):
        """Yields examples."""
        with open(path, "r") as f:
            i = -1
            for line in f:
                article = json.loads(line)
                title = article['title']
                for section_title, section_text in zip(article['section_titles'], article['section_texts']):
                    if section_title in ["See also", "References", "Further reading", "External links"]:
                        continue
                    sentences = section_text.split("\n\n")
                    sentences = [sentence.replace("\n", "") for sentence in sentences if sentence != ""]
                    if len(sentences) < 3:
                        continue
                    if len(sentences) >= 15:
                        continue 
                    mean_len = mean([len(sentence) for sentence in sentences])
                    if mean_len < 30:
                        continue

                    shuffled_sentences, label = self.shuffle_sentences(sentences)
                    i += 1
                    yield i, {
                        _TITLE: title,
                        _SECTION_TITLE: section_title,
                        _SENTENCES: sentences,
                        _SHUFFLED_SENTENCES: shuffled_sentences,
                        _LABEL: label,
                    }

    def shuffle_sentences(self, sentences):
        sentences = np.array(sentences)
        permutation = np.random.permutation(len(sentences))
        return sentences[permutation].tolist(), np.argsort(permutation).tolist()