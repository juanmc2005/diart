<br/>

<p align="center">
<img src="/logo.png" title="Logo" />
</p>

<p align="center">
<img alt="PyPI Version" src="https://img.shields.io/pypi/v/diart?color=g">
<img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/diart?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads">
<img alt="Top language" src="https://img.shields.io/github/languages/top/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="Code size in bytes" src="https://img.shields.io/github/languages/code-size/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="License" src="https://img.shields.io/github/license/juanmc2005/StreamingSpeakerDiarization?color=g">
</p>

<div align="center">
  <h4>
    <a href="#installation">
      Installation
    </a>
    <span> | </span>
    <a href="#stream-audio">
      Stream audio
    </a>
    <span> | </span>
    <a href="#tune-hyper-parameters">
      Tune hyper-parameters
    </a>
    <span> | </span>
    <a href="#build-pipelines">
      Build pipelines
    </a>
    <span> | </span>
    <a href="#powered-by-research">
      Research
    </a>
    <span> | </span>
    <a href="#citation">
      Citation
    </a>
    <span> | </span>
    <a href="#reproducibility">
      Reproducibility
    </a>
  </h4>
</div>

<br/>

<p align="center">
<img width="100%" src="/demo.gif" title="Real-time diarization example" />
</p>

## Installation

1) Create environment:

```shell
conda create -n diart python=3.8
conda activate diart
```

2) Install `PortAudio` and `soundfile`:

```shell
conda install portaudio
conda install pysoundfile -c conda-forge
```

3) [Install PyTorch](https://pytorch.org/get-started/locally/#start-locally)

4) Install pyannote.audio 2.0 (currently in development)

```shell
pip install git+https://github.com/pyannote/pyannote-audio.git@develop#egg=pyannote-audio
```

**Note:** starting from version 0.4, installing pyannote.audio is mandatory to run the default system or to use pyannote-based models. In any other case, this step can be ignored.

5) Install diart:
```shell
pip install diart
```

## Stream audio

### From the command line

A recorded conversation:

```shell
python -m diart.stream /path/to/audio.wav
```

A live conversation:

```shell
python -m diart.stream microphone
```

See `python -m diart.stream -h` for more options.

### From python

Run a real-time speaker diarization pipeline over an audio stream with `RealTimeInference`:

```python
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig

config = PipelineConfig()  # Default parameters
pipeline = OnlineSpeakerDiarization(config)
audio_source = MicrophoneAudioSource(config.sample_rate)
inference = RealTimeInference("/output/path", do_plot=True)
inference(pipeline, audio_source)
```

For faster inference and evaluation on a dataset we recommend to use `Benchmark` instead (see our notes on [reproducibility](#reproducibility)).

## Tune hyper-parameters

Diart implements a hyper-parameter optimizer based on optuna that allows you to tune any pipeline to any dataset.

[More about optuna](https://optuna.readthedocs.io/en/stable/index.html).

### A simple example

```python
from diart.optim import Optimizer, TauActive, RhoUpdate, DeltaNew
from diart.pipelines import PipelineConfig
from diart.inference import Benchmark

# Benchmark runs and evaluates the pipeline on a dataset
benchmark = Benchmark("/wav/dir", "/rttm/dir", "/out/dir", show_report=False)
# Base configuration for the pipeline we're going to tune
base_config = PipelineConfig(duration=5, step=0.5, latency=5)
# Hyper-parameters to optimize
hparams = [TauActive, RhoUpdate, DeltaNew]
# Optimizer implements the optimization loop
optimizer = Optimizer(benchmark, base_config, hparams, "/db/out/dir")
# Run optimization
optimizer.optimize(num_iter=100, show_progress=True)
```

This will store temporary predictions in `/out/dir` and write results of each trial in an sqlite database `/db/out/dir/trials.db`.

### Distributed optimization

For bigger datasets, it is sometimes more convenient to run multiple optimization processes in parallel.
To do this, make sure that the output directory or the study given to the optimizer is the same in each process.
Notice that the output directories of `Benchmark` MUST be different to avoid concurrency issues.

```python
from diart.optim import Optimizer
from diart.inference import Benchmark

ID = 0  # Worker identifier
base_config, hparams = ...
benchmark = Benchmark("/wav/dir", "/rttm/dir", f"/out/dir/worker_{ID}", show_report=False)
optimizer = Optimizer(benchmark, base_config, hparams, "/db/out/dir")
optimizer.optimize(num_iter=100, show_progress=True)
```

It is recommended to use other databases like mysql instead of sqlite in distributed optimization.
More on this [here](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#sphx-glr-tutorial-10-key-features-004-distributed-py).

## Build pipelines

For a more advanced usage, diart also provides building blocks that can be combined to create your own pipeline.
Streaming is powered by [RxPY](https://github.com/ReactiveX/RxPY), but the `blocks` module is completely independent and can be used separately.

### Example

Obtain overlap-aware speaker embeddings from a microphone stream:

```python
import rx
import rx.operators as ops
import diart.operators as dops
from diart.sources import MicrophoneAudioSource
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding
from diart.models import SegmentationModel, EmbeddingModel

# Initialize independent modules
seg_model = SegmentationModel.from_pyannote("pyannote/segmentation")
segmentation = SpeakerSegmentation(seg_model)
emb_model = EmbeddingModel.from_pyannote("pyannote/embedding")
embedding = OverlapAwareSpeakerEmbedding(emb_model)
mic = MicrophoneAudioSource(seg_model.get_sample_rate())

# Reformat microphone stream. Defaults to 5s duration and 500ms shift
regular_stream = mic.stream.pipe(dops.regularize_audio_stream(seg_model.get_sample_rate()))
# Branch the microphone stream to calculate segmentation
segmentation_stream = regular_stream.pipe(ops.map(segmentation))
# Join audio and segmentation stream to calculate speaker embeddings
embedding_stream = rx.zip(
    regular_stream, segmentation_stream
).pipe(ops.starmap(embedding))

embedding_stream.subscribe(on_next=lambda emb: print(emb.shape))

mic.read()
```

Output:

```
torch.Size([4, 512])
torch.Size([4, 512])
torch.Size([4, 512])
...
```

## Powered by research

Diart is the official implementation of the paper *[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](/paper.pdf)* by [Juan Manuel Coria](https://juanmc2005.github.io/), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay](https://saharghannay.github.io/) and [Sophie Rosset](https://perso.limsi.fr/rosset/).


> We propose to address online speaker diarization as a combination of incremental clustering and local diarization applied to a rolling buffer updated every 500ms. Every single step of the proposed pipeline is designed to take full advantage of the strong ability of a recently proposed end-to-end overlap-aware segmentation to detect and separate overlapping speakers. In particular, we propose a modified version of the statistics pooling layer (initially introduced in the x-vector architecture) to give less weight to frames where the segmentation model predicts simultaneous speakers. Furthermore, we derive cannot-link constraints from the initial segmentation step to prevent two local speakers from being wrongfully merged during the incremental clustering step. Finally, we show how the latency of the proposed approach can be adjusted between 500ms and 5s to match the requirements of a particular use case, and we provide a systematic analysis of the influence of latency on the overall performance (on AMI, DIHARD and VoxConverse).

<p align="center">
<img height="400" src="/figure1.png" title="Visual explanation of the system" width="325" />
</p>

## Citation

If you found diart useful, please make sure to cite our paper:

```bibtex
@inproceedings{diart,  
  author={Coria, Juan M. and Bredin, Hervé and Ghannay, Sahar and Rosset, Sophie},  
  booktitle={2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)},   
  title={Overlap-Aware Low-Latency Online Speaker Diarization Based on End-to-End Local Segmentation}, 
  year={2021},
  pages={1139-1146},
  doi={10.1109/ASRU51503.2021.9688044},
}
```

##  Reproducibility

![Results table](/table1.png)

Diart aims to be lightweight and capable of real-time streaming in practical scenarios.
Its performance is very close to what is reported in the paper (and sometimes even a bit better).

To obtain the best results, make sure to use the following hyper-parameters:

| Dataset     | latency | tau    | rho    | delta |
|-------------|---------|--------|--------|-------|
| DIHARD III  | any     | 0.555  | 0.422  | 1.517 |
| AMI         | any     | 0.507  | 0.006  | 1.057 |
| VoxConverse | any     | 0.576  | 0.915  | 0.648 |
| DIHARD II   | 1s      | 0.619  | 0.326  | 0.997 |
| DIHARD II   | 5s      | 0.555  | 0.422  | 1.517 |

`diart.benchmark` and `diart.inference.Benchmark` can quickly run and evaluate the pipeline, and even measure its real-time latency. For instance, for a DIHARD III configuration:

```shell
python -m diart.benchmark /wav/dir --reference /rttm/dir --tau=0.555 --rho=0.422 --delta=1.517 --output /out/dir
```

or using the inference API:

```python
from diart.inference import Benchmark
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig

config = PipelineConfig(
    step=0.5,
    latency=0.5,
    tau_active=0.555,
    rho_update=0.422,
    delta_new=1.517
)
pipeline = OnlineSpeakerDiarization(config)
benchmark = Benchmark("/wav/dir", "/rttm/dir", "/out/dir")

benchmark(pipeline, batch_size=32)
```

This runs a faster inference by pre-calculating model outputs in batches.
See `python -m diart.benchmark -h` for more options.

For convenience and to facilitate future comparisons, we also provide the [expected outputs](/expected_outputs) of the paper implementation in RTTM format for every entry of Table 1 and Figure 5. This includes the VBx offline topline as well as our proposed online approach with latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

![Figure 5](/figure5.png)

##  License

```
MIT License

Copyright (c) 2021 Université Paris-Saclay
Copyright (c) 2021 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<p>Logo generated by <a href="https://www.designevo.com/" title="Free Online Logo Maker">DesignEvo free logo designer</a></p>
