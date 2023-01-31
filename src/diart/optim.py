from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Text, Optional, Union, Any

from optuna import TrialPruned, Study, create_study
from optuna.samplers import TPESampler
from optuna.trial import Trial, FrozenTrial
from tqdm import trange, tqdm

from .audio import FilePath
from .benchmark import Benchmark
from .blocks import BasePipelineConfig, PipelineConfig, OnlineSpeakerDiarization


@dataclass
class HyperParameter:
    name: Text
    low: float
    high: float

    @staticmethod
    def from_name(name: Text) -> 'HyperParameter':
        if name == "tau_active":
            return TauActive
        if name == "rho_update":
            return RhoUpdate
        if name == "delta_new":
            return DeltaNew
        raise ValueError(f"Hyper-parameter '{name}' not recognized")


TauActive = HyperParameter("tau_active", low=0, high=1)
RhoUpdate = HyperParameter("rho_update", low=0, high=1)
DeltaNew = HyperParameter("delta_new", low=0, high=2)


class Optimizer:
    def __init__(
        self,
        speech_path: Union[Text, Path],
        reference_path: Union[Text, Path],
        study_or_path: Union[FilePath, Study],
        batch_size: int = 32,
        pipeline_class: type = OnlineSpeakerDiarization,
        hparams: Optional[Sequence[HyperParameter]] = None,
        base_config: Optional[BasePipelineConfig] = None,
        do_kickstart_hparams: bool = True,
    ):
        self.pipeline_class = pipeline_class
        self.benchmark = Benchmark(
            speech_path,
            reference_path,
            show_progress=True,
            show_report=False,
            batch_size=batch_size,
        )

        self.base_config = base_config
        self.do_kickstart_hparams = do_kickstart_hparams
        if self.base_config is None:
            self.base_config = PipelineConfig()
            self.do_kickstart_hparams = False

        self.hparams = hparams
        if self.hparams is None:
            self.hparams = [TauActive, RhoUpdate, DeltaNew]

        # Make sure hyper-parameters exist in the configuration class given
        possible_hparams = vars(self.base_config)
        for param in self.hparams:
            msg = f"Hyper-parameter {param.name} not found " \
                  f"in configuration {self.base_config.__class__.__name__}"
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
                direction="minimize",
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
        values = {"best_der": study.best_value}
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

        # Instantiate pipeline with the new configuration
        config_class = self.base_config.__class__
        pipeline = self.pipeline_class(config_class(**trial_config))

        # Run pipeline over the dataset
        report = self.benchmark(pipeline)

        # Extract DER from report
        return report.loc["TOTAL", "diarization error rate"]["%"]

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
            self.study.enqueue_trial({
                param.name: getattr(self.base_config, param.name)
                for param in self.hparams
            }, skip_if_exists=True)
        self.study.optimize(self.objective, num_iter, callbacks=[self._callback])
