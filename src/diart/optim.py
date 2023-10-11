from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Text, Optional, Union

from optuna import TrialPruned, Study, create_study
from optuna.samplers import TPESampler
from optuna.trial import Trial, FrozenTrial
from pyannote.metrics.base import BaseMetric
from tqdm import trange, tqdm
from typing_extensions import Literal

from . import blocks
from .audio import FilePath
from .inference import Benchmark


class Optimizer:
    def __init__(
        self,
        pipeline_class: type,
        speech_path: Union[Text, Path],
        reference_path: Union[Text, Path],
        study_or_path: Union[FilePath, Study],
        batch_size: int = 32,
        hparams: Optional[Sequence[blocks.base.HyperParameter]] = None,
        base_config: Optional[blocks.PipelineConfig] = None,
        do_kickstart_hparams: bool = True,
        metric: Optional[BaseMetric] = None,
        direction: Literal["minimize", "maximize"] = "minimize",
    ):
        self.pipeline_class = pipeline_class
        # FIXME can we run this benchmark in parallel?
        #  Currently it breaks the trial progress bar
        self.benchmark = Benchmark(
            speech_path,
            reference_path,
            show_progress=True,
            show_report=False,
            batch_size=batch_size,
        )

        self.metric = metric
        self.direction = direction
        self.base_config = base_config
        self.do_kickstart_hparams = do_kickstart_hparams
        if self.base_config is None:
            self.base_config = self.pipeline_class.get_config_class()()
            self.do_kickstart_hparams = False

        self.hparams = hparams
        if self.hparams is None:
            self.hparams = self.pipeline_class.hyper_parameters()

        # Make sure hyper-parameters exist in the configuration class given
        possible_hparams = vars(self.base_config)
        for param in self.hparams:
            msg = (
                f"Hyper-parameter {param.name} not found "
                f"in configuration {self.base_config.__class__.__name__}"
            )
            assert param.name in possible_hparams, msg

        self._progress: Optional[tqdm] = None

        if isinstance(study_or_path, Study):
            self.study = study_or_path
        elif isinstance(study_or_path, str) or isinstance(study_or_path, Path):
            study_or_path = Path(study_or_path)
            self.study = create_study(
                storage="sqlite:///" + str(study_or_path / f"{study_or_path.stem}.db"),
                sampler=TPESampler(),
                study_name=study_or_path.stem,
                direction=self.direction,
                load_if_exists=True,
            )
        else:
            msg = f"Expected Study object or path-like, but got {type(study_or_path).__name__}"
            raise ValueError(msg)

    @property
    def best_performance(self):
        return self.study.best_value

    @property
    def best_hparams(self):
        return self.study.best_params

    def _callback(self, study: Study, trial: FrozenTrial):
        if self._progress is None:
            return
        self._progress.update(1)
        self._progress.set_description(f"Trial {trial.number + 1}")
        values = {"best_perf": study.best_value}
        for name, value in study.best_params.items():
            values[f"best_{name}"] = value
        self._progress.set_postfix(OrderedDict(values))

    def objective(self, trial: Trial) -> float:
        # Set suggested values for optimized hyper-parameters
        trial_config = vars(self.base_config)
        for hparam in self.hparams:
            trial_config[hparam.name] = trial.suggest_uniform(
                hparam.name, hparam.low, hparam.high
            )

        # Prune trial if required
        if trial.should_prune():
            raise TrialPruned()

        # Instantiate the new configuration for the trial
        config = self.base_config.__class__(**trial_config)

        # Determine the evaluation metric
        metric = self.metric
        if metric is None:
            metric = self.pipeline_class.suggest_metric()

        # Run pipeline over the dataset
        report = self.benchmark(self.pipeline_class, config, metric)

        # Extract target metric from report
        return report.loc["TOTAL", metric.name]["%"]

    def __call__(self, num_iter: int, show_progress: bool = True):
        self._progress = None
        if show_progress:
            self._progress = trange(num_iter)
            last_trial = -1
            if self.study.trials:
                last_trial = self.study.trials[-1].number
            self._progress.set_description(f"Trial {last_trial + 1}")
        # Start with base config hyper-parameters if config was given
        if self.do_kickstart_hparams:
            self.study.enqueue_trial(
                {
                    param.name: getattr(self.base_config, param.name)
                    for param in self.hparams
                },
                skip_if_exists=True,
            )
        self.study.optimize(self.objective, num_iter, callbacks=[self._callback])
