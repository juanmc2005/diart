import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Text, Union, Callable, List, Any

import numpy as np
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
    from whisper.tokenizer import get_tokenizer
    _has_whisper = True
    DecodingResult = whisper.DecodingResult
    DecodingOptions = whisper.DecodingOptions
    Tokenizer = whisper.tokenizer.Tokenizer
except ImportError:
    _has_whisper = False
    DecodingResult = Any
    DecodingOptions = Any
    Tokenizer = Any


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
        fp16: bool = False,
    ) -> 'SpeechRecognitionModel':
        msg = "No whisper-transcribed installation found. " \
              "Visit https://github.com/linto-ai/whisper-timestamped#installation to install"
        assert _has_whisper, msg
        return WhisperSpeechRecognitionModel(name, download_path, in_memory, fp16)

    @property
    def duration(self) -> float:
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        raise NotImplementedError

    def set_language(self, language: Optional[Text] = None):
        raise NotImplementedError

    def set_beam_size(self, size: Optional[int] = None):
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


class WhisperDecoder:
    def __init__(
        self,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1,
    ):
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.temperatures = (0, 0.2, 0.4, 0.6, 0.8, 1)

    @staticmethod
    def get_temperature_options(initial: DecodingOptions, t: float) -> DecodingOptions:
        t_options = {**vars(initial)}
        if t > 0:
            t_options.pop("beam_size", None)
            t_options.pop("patience", None)
        else:
            t_options.pop("best_of", None)
        t_options["temperature"] = t
        return DecodingOptions(**t_options)

    @staticmethod
    def decode(
        model,
        batch: torch.Tensor,
        options: DecodingOptions
    ) -> DecodingResult:
        return model.decode(batch, options)

    def check_compression(self) -> bool:
        return self.compression_ratio_threshold is not None

    def check_logprob(self) -> bool:
        return self.logprob_threshold is not None

    def needs_fallback(self, output: DecodingResult) -> bool:
        if self.check_compression and output.compression_ratio > self.compression_ratio_threshold:
            # Transcription is too repetitive
            return True
        if self.check_logprob and output.avg_logprob < self.logprob_threshold:
            # Average log probability is too low
            return True
        return False

    def decode_with_fallback(
        self,
        model,
        batch: torch.Tensor,
        options: DecodingOptions,
    ) -> DecodingResult:
        batch_size = batch.shape[0]
        results = [None] * batch_size
        retry_idx = torch.ones(batch_size).type(torch.bool)

        for t in self.temperatures:
            # Transcribe with the given temperature
            t_options = self.get_temperature_options(options, t)
            outputs = model.decode(batch[retry_idx], t_options)

            # Determine which outputs need to be transcribed again
            #  based on quality estimates
            output_idx = torch.where(retry_idx)[0]
            for idx, out in zip(output_idx, outputs):
                results[idx] = out
                if not self.needs_fallback(out):
                    retry_idx[idx] = False

            # No output needs fallback, get out of the loop
            if torch.sum(retry_idx).item() == 0:
                break

        return results

    @staticmethod
    def split_with_timestamps(
        result: DecodingResult,
        tokenizer: Tokenizer,
        chunk_duration: float,
        token_duration: float,
    ) -> Transcription:
        tokens = torch.tensor(result.tokens)
        chunks, timestamps = [], []
        ts_tokens = tokens.ge(tokenizer.timestamp_begin)
        single_ts_ending = ts_tokens[-2:].tolist() == [False, True]
        consecutive = torch.where(ts_tokens[:-1] & ts_tokens[1:])[0] + 1
        if len(consecutive) > 0:
            # Output contains two consecutive timestamp tokens
            slices = consecutive.tolist()
            if single_ts_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_pos = sliced_tokens[0].item() - tokenizer.timestamp_begin
                end_pos = sliced_tokens[-1].item() - tokenizer.timestamp_begin
                text_tokens = [token for token in sliced_tokens if token < tokenizer.eot]
                text = tokenizer.decode(text_tokens).strip()
                timestamp = Segment(start_pos * token_duration, end_pos * token_duration)
                if text and timestamp.start != timestamp.end:
                    chunks.append(text)
                    timestamps.append(timestamp)
                last_slice = current_slice
        else:
            duration = chunk_duration
            ts = tokens[ts_tokens.nonzero().flatten()]
            if len(ts) > 0 and ts[-1].item() != tokenizer.timestamp_begin:
                # Use last timestamp as end time for the unique chunk
                last_ts_pos = ts[-1].item() - tokenizer.timestamp_begin
                duration = last_ts_pos * token_duration
            text_tokens = [token for token in tokens if token < tokenizer.eot]
            text = tokenizer.decode(text_tokens).strip()
            if text:
                chunks.append(text)
                timestamps.append(Segment(0, duration))

        return Transcription(result.text, chunks, timestamps)


class WhisperSpeechRecognitionModel(SpeechRecognitionModel):
    def __init__(
        self,
        name: Text,
        download_path: Optional[Union[Text, Path]] = None,
        in_memory: bool = False,
        fp16: bool = False,
        compression_ratio_threshold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1,
    ):
        super().__init__(WhisperLoader(name, download_path, in_memory))
        self.fp16 = fp16
        self.beam_size = None
        self.language = None
        self._token_duration: Optional[float] = None
        self.decoder = WhisperDecoder(compression_ratio_threshold, logprob_threshold)

    @property
    def duration(self) -> float:
        # Whisper's maximum duration per input is 30s
        return whisper.audio.CHUNK_LENGTH

    @property
    def sample_rate(self) -> int:
        return whisper.audio.SAMPLE_RATE

    @property
    def token_duration(self) -> float:
        if self._token_duration is None:
            # 2 mel frames per output token
            input_stride = int(np.rint(whisper.audio.N_FRAMES / self.model.dims.n_audio_ctx))
            # Output token duration is 0.02 seconds
            self._token_duration = input_stride * whisper.audio.HOP_LENGTH / self.sample_rate
        return self._token_duration

    def set_language(self, language: Optional[Text] = None):
        self.language = language

    def set_beam_size(self, size: Optional[int] = None):
        self.beam_size = size

    def forward(self, waveform_batch: torch.Tensor) -> List[Transcription]:
        # Remove channel dimension
        batch = waveform_batch.squeeze(1)
        num_chunk_samples = batch.shape[-1]
        # Compute log mel spectrogram
        batch = whisper.log_mel_spectrogram(batch)
        # Add padding
        dtype = torch.float16 if self.fp16 else torch.float32
        batch = whisper.pad_or_trim(batch, whisper.audio.N_FRAMES).to(batch.device).type(dtype)

        # Transcribe batch
        options = whisper.DecodingOptions(
            task="transcribe",
            language=self.language,
            beam_size=self.beam_size,
            fp16=self.fp16,
        )
        results = self.decoder.decode_with_fallback(self.model, batch, options)
        tokenizer = get_tokenizer(
            self.model.is_multilingual,
            language=options.language,
            task=options.task,
        )

        chunk_duration = int(np.rint(num_chunk_samples / self.sample_rate))
        transcriptions = [
            self.decoder.split_with_timestamps(
                res, tokenizer, chunk_duration, self.token_duration
            )
            for res in results
        ]

        return transcriptions
