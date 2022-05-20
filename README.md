<br/>

<p align="center">
<img src="/logo.png" title="Logo" />
</p>

<p align="center">
<img alt="PyPI" src="https://img.shields.io/pypi/v/diart?color=g">
<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="GitHub" src="https://img.shields.io/github/license/juanmc2005/StreamingSpeakerDiarization?color=g">
</p>

<br/>

<p align="center">
<img width="50%" src="/snippet.png" title="Code snippet" />
</p>
<p align="center">
<img width="100%" src="/visualization.gif" title="Real-time diarization example" />
</p>

## Install

1) Create environment:

```shell
conda create -n diart python=3.8
conda activate diart
```

2) [Install PyTorch](https://pytorch.org/get-started/locally/#start-locally)

3) Install pyannote.audio 2.0 (currently in development)
```shell
pip install git+https://github.com/pyannote/pyannote-audio.git@develop#egg=pyannote-audio
```

4) Install diart:
```shell
pip install diart
```

## Stream your own audio

### A recorded conversation

```shell
python -m diart.stream /path/to/audio.wav
```

### From your microphone

```shell
python -m diart.stream microphone
```

See `python -m diart.stream -h` for more options.

## Inference API

Run a customized real-time speaker diarization pipeline over an audio stream with `diart.inference.RealTimeInference`:

```python
from diart.sources import MicrophoneAudioSource
from diart.inference import RealTimeInference
from diart.pipelines import OnlineSpeakerDiarization, PipelineConfig

pipeline = OnlineSpeakerDiarization(PipelineConfig())
audio_source = MicrophoneAudioSource(pipeline.sample_rate)
inference = RealTimeInference("/output/path", do_plot=True)

inference(pipeline, audio_source)
```

For faster inference and evaluation on a dataset we recommend to use `Benchmark` (see our notes on [reproducibility](#reproducibility))

## Build your own pipeline

Diart also provides building blocks that can be combined to create your own pipeline.
Streaming is powered by [RxPY](https://github.com/ReactiveX/RxPY), but the `blocks` module is completely independent and can be used separately.

### Example

Obtain overlap-aware speaker embeddings from a microphone stream:

```python
import rx
import rx.operators as ops
import diart.operators as dops
from diart.sources import MicrophoneAudioSource
from diart.blocks import FramewiseModel, OverlapAwareSpeakerEmbedding

sample_rate = 16000
mic = MicrophoneAudioSource(sample_rate)

# Initialize independent modules
segmentation = FramewiseModel("pyannote/segmentation")
embedding = OverlapAwareSpeakerEmbedding("pyannote/embedding")

# Reformat microphone stream. Defaults to 5s duration and 500ms shift
regular_stream = mic.stream.pipe(dops.regularize_stream(sample_rate))
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
