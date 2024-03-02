from mmengine.registry import RUNNERS
from mmengine.runner.runner import *
from mmengine.runner.runner import _ParamScheduler
from optuna.integration import TorchDistributedTrial
from optuna.trial import Trial


@RUNNERS.register_module()
class SearchRunner(Runner):
    '''
    Modified version of mmengine.Runner to support distributed trials. \n
    Complete the configuration of searching after self.set_env() is called.
    '''

    cfg: Config

    def __init__(
            self,
            launcher: str = 'none',
            env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
            cfg: Optional[ConfigType] = None,
    ):
        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)

    @classmethod
    def init_first_stage(cls, cfg: ConfigType) -> 'SearchRunner':
        """Build a mmengine runner that is not fully initialized (only set-up
        environment), waiting for optuna configuration.

                Args:
                    cfg (ConfigType): A config used for building runner. Keys of
                        ``cfg`` can see :meth:`__init__`.

                Returns:
                    Runner: A runner build from ``cfg``.
                """
        cfg = copy.deepcopy(cfg)

        runner = cls(
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            cfg=cfg,
        )
        return runner

    def init_second_stage(self,
                          trial: Trial,
                          search_params: Dict,
                          search_controls: Dict
                          ) -> Union[Trial, TorchDistributedTrial]:
        if self._distributed:
            trial = TorchDistributedTrial(trial)

        self.cfg = self.optuna_setup(self.cfg, trial, search_params, search_controls)

        self._lazy_init(
            model=self.cfg['model'],
            work_dir=self.cfg['work_dir'],
            train_dataloader=self.cfg.get('train_dataloader'),
            val_dataloader=self.cfg.get('val_dataloader'),
            test_dataloader=self.cfg.get('test_dataloader'),
            train_cfg=self.cfg.get('train_cfg'),
            val_cfg=self.cfg.get('val_cfg'),
            test_cfg=self.cfg.get('test_cfg'),
            auto_scale_lr=self.cfg.get('auto_scale_lr'),
            optim_wrapper=self.cfg.get('optim_wrapper'),
            param_scheduler=self.cfg.get('param_scheduler'),
            val_evaluator=self.cfg.get('val_evaluator'),
            test_evaluator=self.cfg.get('test_evaluator'),
            default_hooks=self.cfg.get('default_hooks'),
            custom_hooks=self.cfg.get('custom_hooks'),
            data_preprocessor=self.cfg.get('data_preprocessor'),
            load_from=self.cfg.get('load_from'),
            env_cfg=self.cfg.get('env_cfg'),  # type: ignore
            resume=self.cfg.get('resume', False),
            log_processor=self.cfg.get('log_processor'),
            log_level=self.cfg.get('log_level', 'INFO'),
            visualizer=self.cfg.get('visualizer'),
            default_scope=self.cfg.get('default_scope', 'mmengine'),
            randomness=self.cfg.get('randomness', dict(seed=None)),
            experiment_name=self.cfg.get('experiment_name'),
        )

        return trial

    @staticmethod
    def optuna_setup(cfg: Config,
                     trial: Trial,
                     search_params: Dict,
                     search_controls: Dict) -> Config:
        """
        Configure the search parameters.

        Args:
            search_params (dict): The dict contains the hyperparameters to be searched.

        """

        modified_search_params = search_params.copy()
        for k, v in search_params.items():
            if not isinstance(v, (list, tuple)):
                raise ValueError(f"The configuration list should be a list or a tuple"
                                 f"not {type(v)} ({k}).")
            search_type = v[0]

            if search_type == 'float':
                modified_search_params[k] = trial.suggest_float(k, **v[1])
            elif search_type == 'int':
                modified_search_params[k] = trial.suggest_int(k, **v[1])
            elif search_type == 'switch':
                modified_search_params[k] = trial.suggest_categorical(k, **v[1])

        modified_search_params['custom_imports'] = dict(
            imports=['modules'], allow_failed_imports=False)

        _prune = search_controls.get("pruner", None)

        if _prune:
            modified_search_params['custom_hooks'] = [
                dict(type="mmengine.PruneHook")
            ]

        cfg.merge_from_dict(modified_search_params)

        return cfg

    def _lazy_init(self,
                   model: Union[nn.Module, Dict],
                   work_dir: str,
                   train_dataloader: Optional[Union[DataLoader, Dict]] = None,
                   val_dataloader: Optional[Union[DataLoader, Dict]] = None,
                   test_dataloader: Optional[Union[DataLoader, Dict]] = None,
                   train_cfg: Optional[Dict] = None,
                   val_cfg: Optional[Dict] = None,
                   test_cfg: Optional[Dict] = None,
                   auto_scale_lr: Optional[Dict] = None,
                   optim_wrapper: Optional[Union[OptimWrapper, Dict]] = None,
                   param_scheduler: Optional[Union[_ParamScheduler, Dict, List]] = None,
                   val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
                   test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
                   default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
                   custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
                   data_preprocessor: Union[nn.Module, Dict, None] = None,
                   load_from: Optional[str] = None,
                   env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
                   resume: bool = False,
                   log_processor: Optional[Dict] = None,
                   log_level: str = 'INFO',
                   visualizer: Optional[Union[Visualizer, Dict]] = None,
                   default_scope: str = 'mmengine',
                   randomness: Dict = dict(seed=None),
                   experiment_name: Optional[str] = None,
                   ):

        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)
        # lazy initialization
        training_related = [train_dataloader, train_cfg, optim_wrapper]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg, and optim_wrapper should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}, '
                f'optim_wrapper={optim_wrapper}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.optim_wrapper: Optional[Union[OptimWrapper, dict]]
        self.optim_wrapper = optim_wrapper

        self.auto_scale_lr = auto_scale_lr

        # If there is no need to adjust learning rate, momentum or other
        # parameters of optimizer, param_scheduler can be None
        if param_scheduler is not None and self.optim_wrapper is None:
            raise ValueError(
                'param_scheduler should be None when optim_wrapper is None, '
                f'but got {param_scheduler}')

        # Parse `param_scheduler` to a list or a dict. If `optim_wrapper` is a
        # `dict` with single optimizer, parsed param_scheduler will be a
        # list of parameter schedulers. If `optim_wrapper` is
        # a `dict` with multiple optimizers, parsed `param_scheduler` will be
        # dict with multiple list of parameter schedulers.
        self._check_scheduler_cfg(param_scheduler)
        self.param_schedulers = param_scheduler

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()

    @classmethod
    def from_cfg(cls,
                 cfg: ConfigType,
                 ) -> 'SearchRunner':
        """Build a search runner from config. Note that this function
        returns a runner that is not fully initialized.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        runner = cls.init_first_stage(cfg)
        return runner
