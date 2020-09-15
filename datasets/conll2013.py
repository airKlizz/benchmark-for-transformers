# coding=utf-8
# Copyright 2020 HuggingFace nlp Authors.
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
"""TheCoNLL 2013 Dataset."""

from __future__ import absolute_import, division, print_function

import logging

import nlp

import os


_CITATION = """
"""

_DESCRIPTION = """
"""

_URL = "https://drive.google.com/uc?export=download&id=1RYAbisQWhzHWB3RUYBAP0KdN9ofUOwtj"


class CoNLL_2013Config(nlp.BuilderConfig):
    """The CoNLL 2013 Dataset."""

    def __init__(self, **kwargs):
        """BuilderConfig for CoNLL 2013.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(CoNLL_2013Config, self).__init__(**kwargs)


class CoNLL_2013(nlp.GeneratorBasedBuilder):
    """The CoNLL 2013 Dataset."""

    BUILDER_CONFIGS = [
        CoNLL_2013Config(
            name="conll2013", version=nlp.Version("1.0.0"), description="The CoNLL 2013 dataset"
        ),
    ]

    def _info(self):
        return nlp.DatasetInfo(
            description=_DESCRIPTION,
            features=nlp.Features(
                {
                    "id": nlp.Value("string"),
                    "tokens": nlp.Sequence(nlp.Value("string")),
                    "labels": nlp.Sequence(nlp.Value("string")),
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_path = dl_manager.download_and_extract(_URL)

        return [
            nlp.SplitGenerator(name=nlp.Split.TRAIN, gen_kwargs={"filepath": os.path.join(data_path, "eng.train.conll")}),
            nlp.SplitGenerator(name=nlp.Split.VALIDATION, gen_kwargs={"filepath": os.path.join(data_path, "eng.testa.conll")}),
            nlp.SplitGenerator(name=nlp.Split.TEST, gen_kwargs={"filepath": os.path.join(data_path, "eng.testb.conll")}),
        ]

    def _generate_examples(self, filepath):
        logging.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            current_tokens = []
            current_labels = []
            sentence_counter = 0
            for row in f:
                row = row.rstrip()
                if row:
                    token, label = row.split("\t")
                    current_tokens.append(token)
                    current_labels.append(label)
                else:
                    # New sentence
                    if not current_tokens:
                        # Consecutive empty lines will cause empty sentences
                        continue
                    assert len(current_tokens) == len(current_labels), "💔 between len of tokens & labels"
                    sentence = (
                        sentence_counter,
                        {
                            "id": str(sentence_counter),
                            "tokens": current_tokens,
                            "labels": current_labels,
                        },
                    )
                    sentence_counter += 1
                    current_tokens = []
                    current_labels = []
                    yield sentence
            # Don't forget last sentence in dataset 🧐
            if current_tokens:
                yield sentence_counter, {
                    "id": str(sentence_counter),
                    "tokens": current_tokens,
                    "labels": current_labels,
                }
