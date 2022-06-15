from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Text, Optional

from optuna import TrialPruned, Study, create_study
from optuna.pruners import BasePruner
from optuna.samplers import TPESampler, BaseSampler
from optuna.trial import Trial, FrozenTrial
from tqdm import trange, tqdm

from .benchmark import Benchmark
from .pipelines import PipelineConfig, OnlineSpeakerDiarization


@dataclass
class HyperParameter:
    name: Text
    low: float
    high: float


TauActive = HyperParameter("tau_active", low=0, high=1)
RhoUpdate = HyperParameter("rho_update", low=0, high=1)
DeltaNew = HyperParameter("delta_new", low=0, high=2)


class OptimizationObjective:
    def __init__(
        self,
        benchmark: Benchmark,
        base_config: PipelineConfig,
        hparams: Iterable[HyperParameter],
    ):
        self.benchmark = benchmark
        self.base_config = base_config
        self.hparams = hparams

    def __call__(self, trial: Trial) -> float:
        # Set suggested values for optimized hyper-parameters
        trial_config = vars(self.base_config)
        for hparam in self.hparams:
            trial_config[hparam.name] = trial.suggest_uniform(
                hparam.name, hparam.low, hparam.high
            )

        # Instantiate pipeline with the new configuration
        pipeline = OnlineSpeakerDiarization(PipelineConfig(**trial_config))

        # Prune trial if required
        if trial.should_prune():
            raise TrialPruned()

        # Run pipeline over the dataset
        report = self.benchmark(pipeline)

        # Clean RTTM files
        for tmp_file in self.benchmark.output_path.iterdir():
            if tmp_file.name.endswith(".rttm"):
                tmp_file.unlink()

        # Extract DER from report
        return report.loc["TOTAL", "diarization error rate"]["%"]


class Optimizer:
    def __init__(
        self,
        objective: OptimizationObjective,
        study_name: Optional[Text] = None,
        storage: Optional[Text] = None,
        sampler: Optional[BaseSampler] = None,
        pruner: Optional[BasePruner] = None,
    ):
        self.objective = objective
        self.study = create_study(
            storage=self.default_storage if storage is None else storage,
            sampler=TPESampler() if sampler is None else sampler,
            pruner=pruner,
            study_name=self.default_study_name if study_name is None else study_name,
            direction="minimize",
            load_if_exists=True,
        )
        self._progress: Optional[tqdm] = None

    @property
    def default_output_path(self) -> Path:
        return self.objective.benchmark.output_path.parent

    @property
    def default_study_name(self) -> Text:
        return self.default_output_path.name

    @property
    def default_storage(self) -> Text:
        return "sqlite:///" + str(self.default_output_path / "trials.db")

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

    def optimize(self, num_iter: int, show_progress: bool = True):
        self._progress = None
        if show_progress:
            self._progress = trange(num_iter)
            last_trial = self.study.trials[-1].number
            self._progress.set_description(f"Trial {last_trial + 1}")
        self.study.optimize(self.objective, num_iter, callbacks=[self._callback])
