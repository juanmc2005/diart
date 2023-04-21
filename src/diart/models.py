from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Text, Union, Callable, List, Tuple, Dict

import torch
import torch.nn as nn
from pyannote.core import Segment

try:
    import pyannote.audio.pipelines.utils as pyannote_loader
    _has_pyannote = True
except ImportError:
    _has_pyannote = False

try:
    import whisper
    _has_whisper = True
except ImportError:
    _has_whisper = False


class PyannoteLoader:
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__()
        self.model_info = model_info
        self.hf_token = hf_token

    def __call__(self) -> nn.Module:
        return pyannote_loader.get_model(self.model_info, self.hf_token)


class WhisperLoader:
    def __init__(
        self,
        name: Text,
        download_path: Optional[Union[Text, Path]] = None,
        in_memory: bool = False,
    ):
        self.name = name
        self.download_path = download_path
        self.in_memory = in_memory

    def __call__(self) -> nn.Module:
        return whisper.load_model(
            name=self.name,
            device="cpu",
            download_root=self.download_path,
            in_memory=self.in_memory,
        )


class LazyModel(nn.Module):
    def __init__(self, loader: Callable[[], nn.Module]):
        super().__init__()
        self.get_model = loader
        self.model: Optional[nn.Module] = None

    def is_in_memory(self) -> bool:
        """Return whether the model has been loaded into memory"""
        return self.model is not None

    def load(self):
        if not self.is_in_memory():
            self.model = self.get_model()

    def to(self, *args, **kwargs) -> nn.Module:
        self.load()
        return super().to(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.load()
        return super().__call__(*args, **kwargs)


class SegmentationModel(LazyModel):
    """
    Minimal interface for a segmentation model.
    """
    @staticmethod
    def from_pyannote(model, use_hf_token: Union[Text, bool, None] = True) -> 'SegmentationModel':
        """
        Returns a `SegmentationModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: SegmentationModel
        """
        assert _has_pyannote, "No pyannote.audio installation found"
        return PyannoteSegmentationModel(model, use_hf_token)

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    @property
    def duration(self) -> float:
        raise NotImplementedError

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the segmentation model.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)

        Returns
        -------
        speaker_segmentation: torch.Tensor, shape (batch, frames, speakers)
        """
        raise NotImplementedError


class PyannoteSegmentationModel(SegmentationModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))

    @property
    def sample_rate(self) -> int:
        self.load()
        return self.model.audio.sample_rate

    @property
    def duration(self) -> float:
        self.load()
        return self.model.specifications.duration

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.model(waveform)


class EmbeddingModel(LazyModel):
    """Minimal interface for an embedding model."""
    @staticmethod
    def from_pyannote(model, use_hf_token: Union[Text, bool, None] = True) -> 'EmbeddingModel':
        """
        Returns an `EmbeddingModel` wrapping a pyannote model.

        Parameters
        ----------
        model: pyannote.PipelineModel
            The pyannote.audio model to fetch.
        use_hf_token: str | bool, optional
            The Huggingface access token to use when downloading the model.
            If True, use huggingface-cli login token.
            Defaults to None.

        Returns
        -------
        wrapper: EmbeddingModel
        """
        assert _has_pyannote, "No pyannote.audio installation found"
        return PyannoteEmbeddingModel(model, use_hf_token)

    def forward(
        self,
        waveform: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of an embedding model with optional weights.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
        weights: Optional[torch.Tensor], shape (batch, frames)
            Temporal weights for each sample in the batch. Defaults to no weights.

        Returns
        -------
        speaker_embeddings: torch.Tensor, shape (batch, embedding_dim)
        """
        raise NotImplementedError


class PyannoteEmbeddingModel(EmbeddingModel):
    def __init__(self, model_info, hf_token: Union[Text, bool, None] = True):
        super().__init__(PyannoteLoader(model_info, hf_token))

    def forward(
        self,
        waveform: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(waveform, weights=weights)


@dataclass(frozen=True)
class Transcription:
    text: Text
    chunks: List[Text]
    timestamps: List[Segment]


class SpeechRecognitionModel(LazyModel):
    @staticmethod
    def from_whisper(
        name: Text,
        download_path: Optional[Union[Text, Path]] = None,
        in_memory: bool = False,
        remember_transcriptions: bool = True,
        fp16: bool = False,
    ) -> 'SpeechRecognitionModel':
        msg = "No whisper-transcribed installation found. " \
              "Visit https://github.com/linto-ai/whisper-timestamped#installation to install"
        assert _has_whisper, msg
        return WhisperSpeechRecognitionModel(
            name, download_path, in_memory, remember_transcriptions, fp16
        )

    @property
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    def set_language(self, language: Optional[Text] = None):
        raise NotImplementedError

    def forward(self, waveform: torch.Tensor) -> List[Transcription]:
        """
        Forward pass of the speech recognition model.

        Parameters
        ----------
        waveform: torch.Tensor, shape (batch, channels, samples)
            Batch of audio chunks to transcribe

        Returns
        -------
        transcriptions: List[Transcription]
            A list of timestamped transcriptions
        """
        raise NotImplementedError


class WhisperSpeechRecognitionModel(SpeechRecognitionModel):
    def __init__(
        self,
        name: Text,
        download_path: Optional[Union[Text, Path]] = None,
        in_memory: bool = False,
        remember_transcriptions: bool = True,
        fp16: bool = False,
    ):
        super().__init__(WhisperLoader(name, download_path, in_memory))
        self.remember_transcriptions = remember_transcriptions
        self.fp16 = fp16
        self.language = None
        self._cache = None

    @property
    def duration(self) -> float:
        # Whisper's maximum duration per input is 30s
        return whisper.audio.CHUNK_LENGTH

    @property
    def sample_rate(self) -> int:
        return whisper.audio.SAMPLE_RATE

    def set_language(self, language: Optional[Text] = None):
        self.language = language

    def forward(self, waveform_batch: torch.Tensor) -> List[Transcription]:
        results = []
        for waveform in waveform_batch:
            audio = whisper.pad_or_trim(waveform.type(torch.float32).reshape(-1))
            transcription = whisper.transcribe(
                self.model,
                audio,
                initial_prompt=self._cache,
                verbose=None,
                task="transcribe",
                language=self.language,
                fp16=self.fp16,
            )

            # Extract chunks and timestamps
            chunks, timestamps = [], []
            for chunk in transcription["segments"]:
                chunks.append(chunk["text"])
                timestamps.append(Segment(chunk["start"], chunk["end"]))

            # Create transcription object
            transcription = Transcription(transcription["text"], chunks, timestamps)
            results.append(transcription)

            # Update transcription buffer
            if self.remember_transcriptions:
                # TODO handle overlapping transcriptions
                self._cache = transcription.text

        return results
