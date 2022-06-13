from pathlib import Path
from typing import Dict, Text, Tuple, Optional, Callable

import optuna
from optuna.pruners._base import BasePruner
from optuna.samplers import TPESampler
from optuna.samplers._base import BaseSampler
from optuna.trial import Trial

from audio import FilePath
from benchmark import Benchmark
from pipelines import PipelineConfig, OnlineSpeakerDiarization


class HyperParameterOptimizer:
    def __init__(
        self,
        speech_path: FilePath,
        reference_path: FilePath,
        output_path: FilePath,
        base_config: PipelineConfig,
        hparams: Dict[Text, Tuple[float, float]],
        batch_size: int = 32,
    ):
        self.base_config = base_config
        self.hparams = hparams
        self.batch_size = batch_size
        self.output_path = Path(output_path).expanduser()
        self.output_path.mkdir(parents=True, exist_ok=False)
        self.tmp_path = self.output_path / "current_iter"
        self.benchmark = Benchmark(speech_path, reference_path, self.tmp_path)

    def _objective(self, trial: Trial) -> float:
        # Set suggested values for optimized hyper-parameters
        trial_config = vars(self.base_config)
        for hp_name, (low, high) in self.hparams.items():
            trial_config[hp_name] = trial.suggest_uniform(hp_name, low, high)

        # Instantiate pipeline with the new configuration
        pipeline = OnlineSpeakerDiarization(PipelineConfig(**trial_config))

        # Run pipeline over the dataset
        report = self.benchmark(pipeline, self.batch_size, verbose=False)

        # Clean RTTM files
        for tmp_file in self.tmp_path.iterdir():
            tmp_file.unlink()

        # Extract DER from report
        return report.loc["TOTAL", "diarization error rate %"]

    def optimize(
        self,
        sampler: Optional[BaseSampler] = None,
        pruner: Optional[BasePruner] = None,
        num_iter: int = 100,
        experiment_name: Optional[Text] = None,
    ) -> Tuple[float, Dict[Text, float]]:
        """Optimize the given hyper-parameters on the given dataset.

        Parameters
        ----------
        sampler: Optional[optuna.BaseSampler]
            The Optuna sampler to use during optimization. Defaults to TPESampler.
        pruner: Optional[optuna.BasePruner]
            The Optuna pruner to use during optimization. Defaults to None.
        num_iter: int
            Number of iterations over the dataset. Defaults to 100.
        experiment_name: Optional[Text]
            Name of the optimization run. Defaults to None.

        Returns
        -------
        der_value: float
            Diarization error rate of the best iteration.
        best_hparams: Dict[Text, float]
            Hyper-parameters of the best iteration.
        """
        sampler = TPESampler() if sampler is None else sampler
        storage = self.output_path / "trials.db"
        study = optuna.create_study(
            storage=f"sqlite://{str(storage)}",
            sampler=sampler,
            pruner=pruner,
            study_name=experiment_name,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(self._objective, n_trials=num_iter)
        return study.best_value, study.best_params
