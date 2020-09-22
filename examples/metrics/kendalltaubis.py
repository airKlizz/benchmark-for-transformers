# coding=utf-8
# Copyright 2020 The HuggingFace NLP Authors.
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
""" Kendall's tau by hand. """

import absl  # Here to have a nice missing dependency error message early on
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import six  # Here to have a nice missing dependency error message early on

_CITATION = """
"""

_DESCRIPTION = """\
Let ( x 1 , y 1 ) , . . . , ( x n , y n ) {\displaystyle (x_{1},y_{1}),...,(x_{n},y_{n})} {\displaystyle (x_{1},y_{1}),...,(x_{n},y_{n})} be a set of observations of the joint random variables X and Y, such that all the values of ( x i {\displaystyle x_{i}} x_{i}) and ( y i {\displaystyle y_{i}} y_{i}) are unique (ties are neglected for simplicity). Any pair of observations ( x i , y i ) {\displaystyle (x_{i},y_{i})} {\displaystyle (x_{i},y_{i})} and ( x j , y j ) {\displaystyle (x_{j},y_{j})} {\displaystyle (x_{j},y_{j})}, where i < j {\displaystyle i<j} i<j, are said to be concordant if the sort order of ( x i , x j ) {\displaystyle (x_{i},x_{j})} {\displaystyle (x_{i},x_{j})} and ( y i , y j ) {\displaystyle (y_{i},y_{j})} {\displaystyle (y_{i},y_{j})} agrees: that is, if either both x i > x j {\displaystyle x_{i}>x_{j}} {\displaystyle x_{i}>x_{j}} and y i > y j {\displaystyle y_{i}>y_{j}} {\displaystyle y_{i}>y_{j}} holds or both x i < x j {\displaystyle x_{i}<x_{j}} {\displaystyle x_{i}<x_{j}} and y i < y j {\displaystyle y_{i}<y_{j}} {\displaystyle y_{i}<y_{j}}; otherwise they are said to be discordant.

The Kendall τ coefficient is defined as:

    τ = ( number of concordant pairs ) − ( number of discordant pairs ) ( n 2 ) . {\displaystyle \tau ={\frac {({\text{number of concordant pairs}})-({\text{number of discordant pairs}})}{n \choose 2}}.} {\displaystyle \tau ={\frac {({\text{number of concordant pairs}})-({\text{number of discordant pairs}})}{n \choose 2}}.}[3]

Where ( n 2 ) = n ( n − 1 ) 2 {\displaystyle {n \choose 2}={n(n-1) \over 2}} {\displaystyle {n \choose 2}={n(n-1) \over 2}} is the binomial coefficient for the number of ways to choose two items from n items. 

https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
"""

_KWARGS_DESCRIPTION = """
Calculate Kendall’s tau, a correlation measure for ordinal data.
Args:
    predictions: list of predictions to score. Each predictions
        should be a list of list of rankings.
    references: list of reference for each prediction. Each predictions
        should be a list of list of rankings.
Returns:
    tau: The tau statistic,
"""

def substitution(X, Y):

    assert len(X) == len(Y)

    permutation = {}
    for i, x in enumerate(X):
        permutation[x] = i
    for i in range(len(Y)):
        Y[i] = permutation[Y[i]]

    return Y

def get_nb_inv(X): 
  
    nb_inv = 0
    for i in range(len(X)): 
        for j in range(i + 1, len(X)): 
            if (X[i] > X[j]): 
                nb_inv += 1
  
    return nb_inv 

def kendall_tau_bis(X, Y):

    new_Y = substitution(X, Y)
    nb_inv = get_nb_inv(new_Y)
    n = len(X)
    binomial_coefficient = n*(n-1)/2
    tau = 1 - 2*nb_inv/binomial_coefficient

    return tau

class KendallTauBis(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int8")),
                    "references": datasets.Sequence(datasets.Value("int8")),
                }
            ),
            codebase_urls=[""],
            reference_urls=["https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient"],
        )

    def _compute(self, predictions, references, initial_lexsort=None, nan_policy="propagate", method="auto"):
        result = {"tau": np.array([])}

        for prediction, reference in zip(predictions, references):
            tau = kendall_tau_bis(
                X=prediction, Y=reference
            )
            result["tau"] = np.append(result["tau"], tau)

        result["tau"] = result["tau"].mean()
        return result