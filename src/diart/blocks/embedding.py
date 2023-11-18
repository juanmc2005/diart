from typing import Optional, Union, Text

import torch
from einops import rearrange

from .. import functional as F
from ..features import TemporalFeatures, TemporalFeatureFormatter
from ..models import EmbeddingModel


class SpeakerEmbedding:
    def __init__(self, model: EmbeddingModel, device: Optional[torch.device] = None):
        self.model = model
        self.model.eval()
        self.device = device
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.waveform_formatter = TemporalFeatureFormatter()
        self.weights_formatter = TemporalFeatureFormatter()

    @staticmethod
    def from_pretrained(
        model,
        use_hf_token: Union[Text, bool, None] = True,
        device: Optional[torch.device] = None,
    ) -> "SpeakerEmbedding":
        emb_model = EmbeddingModel.from_pretrained(model, use_hf_token)
        return SpeakerEmbedding(emb_model, device)

    def __call__(
        self, waveform: TemporalFeatures, weights: Optional[TemporalFeatures] = None
    ) -> torch.Tensor:
        """
        Calculate speaker embeddings of input audio.
        If weights are given, calculate many speaker embeddings from the same waveform.

        Parameters
        ----------
        waveform: TemporalFeatures, shape (samples, channels) or (batch, samples, channels)
        weights: Optional[TemporalFeatures], shape (frames, speakers) or (batch, frames, speakers)
            Per-speaker and per-frame weights. Defaults to no weights.

        Returns
        -------
        embeddings: torch.Tensor
            If weights are provided, the shape is (batch, speakers, embedding_dim),
            otherwise the shape is (batch, embedding_dim).
            If batch size == 1, the batch dimension is omitted.
        """
        with torch.no_grad():
            inputs = self.waveform_formatter.cast(waveform).to(self.device)
            inputs = rearrange(inputs, "batch sample channel -> batch channel sample")
            if weights is not None:
                weights = self.weights_formatter.cast(weights).to(self.device)
                batch_size, _, num_speakers = weights.shape
                inputs = inputs.repeat(1, num_speakers, 1)
                weights = rearrange(weights, "batch frame spk -> (batch spk) frame")
                inputs = rearrange(inputs, "batch spk sample -> (batch spk) 1 sample")
                output = rearrange(
                    self.model(inputs, weights),
                    "(batch spk) feat -> batch spk feat",
                    batch=batch_size,
                    spk=num_speakers,
                )
            else:
                output = self.model(inputs)
            return output.squeeze().cpu()


class OverlappedSpeechPenalty:
    """Applies a penalty on overlapping speech and low-confidence regions to speaker segmentation scores.

    .. note::
        For more information, see `"Overlap-Aware Low-Latency Online Speaker Diarization
        based on End-to-End Local Segmentation" <https://github.com/juanmc2005/diart/blob/main/paper.pdf>`_
        (Section 2.2.1 Segmentation-driven speaker embedding). This block implements Equation 2.

    Parameters
    ----------
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    normalize: bool, optional
        Whether to min-max normalize weights to be in the range [0, 1].
        Defaults to False.
    """

    def __init__(self, gamma: float = 3, beta: float = 10, normalize: bool = False):
        self.gamma = gamma
        self.beta = beta
        self.formatter = TemporalFeatureFormatter()
        self.normalize = normalize

    def __call__(self, segmentation: TemporalFeatures) -> TemporalFeatures:
        weights = self.formatter.cast(segmentation)  # shape (batch, frames, speakers)
        with torch.inference_mode():
            weights = F.overlapped_speech_penalty(weights, self.gamma, self.beta)
            if self.normalize:
                min_values = weights.min(dim=1, keepdim=True).values
                max_values = weights.max(dim=1, keepdim=True).values
                weights = (weights - min_values) / (max_values - min_values)
                weights.nan_to_num_(1e-8)
        return self.formatter.restore_type(weights)


class EmbeddingNormalization:
    def __init__(self, norm: Union[float, torch.Tensor] = 1):
        self.norm = norm
        # Add batch dimension if missing
        if isinstance(self.norm, torch.Tensor) and self.norm.ndim == 2:
            self.norm = self.norm.unsqueeze(0)

    def __call__(self, embeddings: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            norm_embs = F.normalize_embeddings(embeddings, self.norm)
        return norm_embs


class OverlapAwareSpeakerEmbedding:
    """
    Extract overlap-aware speaker embeddings given an audio chunk and its segmentation.

    Parameters
    ----------
    model: EmbeddingModel
        A pre-trained embedding model.
    gamma: float, optional
        Exponent to lower low-confidence predictions.
        Defaults to 3.
    beta: float, optional
        Softmax's temperature parameter (actually 1/beta) to lower joint speaker activations.
        Defaults to 10.
    norm: float or torch.Tensor of shape (batch, speakers, 1) where batch is optional
        The target norm for the embeddings. It can be different for each speaker.
        Defaults to 1.
    normalize_weights: bool, optional
        Whether to min-max normalize embedding weights to be in the range [0, 1].
    device: Optional[torch.device]
        The device on which to run the embedding model.
        Defaults to GPU if available or CPU if not.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        self.embedding = SpeakerEmbedding(model, device)
        self.osp = OverlappedSpeechPenalty(gamma, beta, normalize_weights)
        self.normalize = EmbeddingNormalization(norm)

    @staticmethod
    def from_pretrained(
        model,
        gamma: float = 3,
        beta: float = 10,
        norm: Union[float, torch.Tensor] = 1,
        use_hf_token: Union[Text, bool, None] = True,
        normalize_weights: bool = False,
        device: Optional[torch.device] = None,
    ):
        model = EmbeddingModel.from_pretrained(model, use_hf_token)
        return OverlapAwareSpeakerEmbedding(
            model, gamma, beta, norm, normalize_weights, device
        )

    def __call__(
        self, waveform: TemporalFeatures, segmentation: TemporalFeatures
    ) -> torch.Tensor:
        return self.normalize(self.embedding(waveform, self.osp(segmentation)))
