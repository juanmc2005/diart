import torch

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which breaks loading
# older pyannote checkpoints that contain omegaconf and lightning objects.
# Register the required globals so weights_only=True can still deserialize them.
if hasattr(torch.serialization, "add_safe_globals"):
    from collections import defaultdict
    from typing import Any

    import pyannote.audio
    import pytorch_lightning
    from omegaconf import DictConfig, ListConfig
    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.nodes import AnyNode

    torch.serialization.add_safe_globals(
        [
            Any,
            AnyNode,
            ContainerMetadata,
            defaultdict,
            dict,
            int,
            list,
            DictConfig,
            ListConfig,
            Metadata,
            torch.torch_version.TorchVersion,
            pyannote.audio.core.model.Introspection,
            pyannote.audio.core.task.Problem,
            pyannote.audio.core.task.Resolution,
            pyannote.audio.core.task.Specifications,
            pytorch_lightning.callbacks.early_stopping.EarlyStopping,
            pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint,
        ]
    )

from .blocks import (
    SpeakerDiarization,
    Pipeline,
    SpeakerDiarizationConfig,
    PipelineConfig,
    VoiceActivityDetection,
    VoiceActivityDetectionConfig,
)
