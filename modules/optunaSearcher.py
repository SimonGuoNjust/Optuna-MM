import os
from pathlib import Path
from typing import Callable

import optuna
from mmengine.fileio.io import load
from optuna.pruners import *
from optuna.samplers import *
from optuna.study import create_study, Study
from optuna.trial import FrozenTrial


class Searcher:
    sampler_mapping = {
        "TPE": TPESampler,
        "Random": RandomSampler,
    }

    pruner_mapping = {
        "Hyperband": HyperbandPruner,
        "Median": MedianPruner,
    }

    def __init__(self,
                 optuna_cfg: Path,
                 ):
        self.cfg_path = optuna_cfg
        self.optuna_cfg = load(optuna_cfg)
        self.study = None
        self.num_trials = self.optuna_cfg.get("trails", 5)
        self.gpus = self.optuna_cfg.get("gpus", 0)

    def create_study(self):
        sampler_cfg = self.optuna_cfg.get('sampler', None)
        pruner_cfg = self.optuna_cfg.get('pruner', None)

        self.study = create_study(
            study_name=self.optuna_cfg.get('name', 'study'),
            direction=self.optuna_cfg.get('direction', 'maximize'),
            sampler=self.sampler_mapping[sampler_cfg['type']](
                **sampler_cfg['args']) if sampler_cfg else RandomSampler(),
            pruner=self.pruner_mapping[pruner_cfg['type']](**pruner_cfg['args']) if pruner_cfg else None,
        )

    def optuna_cb(self, study: Study, trial: FrozenTrial):
        print(trial.params)
        # if trial.value >= study.best_value:
        #     work_dir = trial.user_attrs.get('work_dir', None)
        #     if work_dir is not None:
        #         ckpt = os.path.join(work_dir, "epoch_12.pth")
        #         if not os.path.exists(ckpt):
        #             return
        #         ckpt_dst = os.path.join(work_dir, "best.pth")
        #         shutil.copy(ckpt, ckpt_dst)
        #         print(f"Save trial {trial._trial_id} checkpoint as best")

    def search(self,
               train: Callable,
               local_rank: int = 0,
               save_dir: Path = None):
        self.create_study()
        if self.gpus > 1:
            if local_rank == 0:
                self.study.optimize(train, n_trials=self.num_trials, )
                self.summary(save_dir)
            else:
                for _ in range(self.num_trials):
                    try:
                        train(None)
                    except optuna.exceptions.TrialPruned:
                        pass
        else:
            self.study.optimize(train, n_trials=self.num_trials, callbacks=[self.optuna_cb])
            self.summary(save_dir)

    def summary(self,
                save_dir: Path = None):

        pruned_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        if save_dir is not None:
            dataframe = self.study.trials_dataframe()
            dataframe.to_csv(os.path.join(save_dir, 'results.csv'))
