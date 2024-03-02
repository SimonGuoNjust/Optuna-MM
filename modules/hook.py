from typing import List, Callable, Optional, Dict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class PruneHook(Hook):
    def __init__(self,
                 metrics: List[str],
                 score_func: Callable[[List[float]], float]):
        self._metrics = metrics
        self._score_func = score_func

    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:

        """Decide whether to stop the training process.

        Args:
            runner (Runner): The runner of the training process.
            metrics (dict): Evaluation results of all metrics
        """

        metric_values = []
        for metric in self._metrics:
            value = metrics.get(metric, None)
            if value is None:
                runner.train_loop.stop_training = True
                runner.cfg._trial.report(0, runner.epoch)
                return
            metric_values.append(value)

        score = self._score_func(metric_values)
        runner.cfg._trial.report(score, runner.epoch)
        if runner.cfg._trial.should_prune():
            runner.train_loop.stop_training = True
