from typing import Dict, Union, List, Tuple

import numpy as np


def oneMetric(metrics: Dict,
              name: str) -> float:
    if name not in metrics.keys():
        raise ValueError(f"Metric {name} is not reported!")
    return metrics[name]


def metricsWeightedAverage(metrics: Dict,
                           weights: Union[List, Tuple, np.ndarray, Dict]) -> float:
    score = 0.0
    if isinstance(weights, Dict):
        for key in weights:
            if key not in metrics.keys():
                raise ValueError(f"Metric {key} is not reported!")
        metric_weights = weights
    else:
        assert len(weights) == len(metrics.keys()), \
            f"The number of weights({len(weights)}) must be equal to" \
            f"the number of metrics({len(metrics.keys())}). "
        metric_weights = dict()
        for weight, key in zip(weights, metrics.keys()):
            metric_weights[key] = weight

    for key, value in metrics.items():
        if np.isnan(value):
            raise ValueError(f"Metric {key} is Nan.")
        score += metric_weights[key] * value

    return score


def metricsAverage(metrics: Dict,
                   names: List[str]) -> float:
    metric_weights = dict()
    number = len(names)
    for key in metrics.keys():
        metric_weights[key] = 1 / number if key in names else 0

    return metricsWeightedAverage(metrics, metric_weights)


def metricsSum(metrics: Dict,
               names: List[str]) -> float:
    metric_weights = dict()
    for key in metrics.keys():
        metric_weights[key] = 1 if key in names else 0

    return metricsWeightedAverage(metrics, metric_weights)


score_funcs = {
    "Specific": oneMetric,
    "Average": metricsAverage,
    "Sum": metricsSum,
    "Weighted": metricsWeightedAverage,
}
