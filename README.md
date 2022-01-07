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
<img width="48%" src="/snippet.png" title="Code snippet" />
<img width="51%" src="/visualization.gif" title="Real-time diarization example" />
</p>

## Demo

You can visualize the real-time speaker diarization of an audio stream with the built-in demo script.

### Stream a recorded conversation

```shell
python -m diart.demo /path/to/audio.wav
```

### Stream from your microphone

```shell
python -m diart.demo microphone
```

See `python -m diart.demo -h` for more information.

## Build your own pipeline

Diart provides building blocks that can be combined to do speaker diarization on an audio stream.
The streaming implementation is powered by [RxPY](https://github.com/ReactiveX/RxPY), but the `functional` module is completely independent.

### Example

Obtain overlap-aware speaker embeddings from a microphone stream

```python
import rx
import rx.operators as ops
import diart.operators as myops
from diart.sources import MicrophoneAudioSource
import diart.functional as fn

sample_rate = 16000
mic = MicrophoneAudioSource(sample_rate)

# Initialize independent modules
segmentation = fn.FrameWiseModel("pyannote/segmentation")
embedding = fn.ChunkWiseModel("pyannote/embedding")
osp = fn.OverlappedSpeechPenalty(gamma=3, beta=10)
normalization = fn.EmbeddingNormalization(norm=1)

# Reformat microphone stream. Defaults to 5s duration and 500ms shift
regular_stream = mic.stream.pipe(myops.regularize_stream(sample_rate))
# Branch the microphone stream to calculate segmentation
segmentation_stream = regular_stream.pipe(ops.map(segmentation))
# Join audio and segmentation stream to calculate speaker embeddings
embedding_stream = rx.zip(regular_stream, segmentation_stream).pipe(
    ops.starmap(lambda wave, seg: (wave, osp(seg))),
    ops.starmap(embedding),
    ops.map(normalization)
)

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

## Install

1) Create environment:

```shell
conda create -n diarization python==3.8
conda activate diarization
```

2) Install the latest PyTorch version following the [official instructions](https://pytorch.org/get-started/locally/#start-locally)

3) Install pyannote.audio 2.0 (currently in development)
```shell
pip install git+https://github.com/pyannote/pyannote-audio.git@develop#egg=pyannote-audio
```

4) Install diart:
```shell
pip install diart
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
Awaiting paper publication (ASRU 2021).
```

##  Reproducibility

![Results table](/table1.png)

To reproduce the results of the paper, use the following hyper-parameters:

Dataset     | latency | tau    | rho    | delta 
------------|---------|--------|--------|------
DIHARD III  | any     | 0.555  | 0.422  | 1.517  
AMI         | any     | 0.507  | 0.006  | 1.057  
VoxConverse | any     | 0.576  | 0.915  | 0.648  
DIHARD II   | 1s      | 0.619  | 0.326  | 0.997  
DIHARD II   | 5s      | 0.555  | 0.422  | 1.517

For instance, for a DIHARD III configuration:

```shell
python -m diart.demo /path/to/file.wav --tau=0.555 --rho=0.422 --delta=1.517 --output /output/dir
```

And then to obtain the diarization error rate:

```python
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.database.util import load_rttm

metric = DiarizationErrorRate()
hypothesis = load_rttm("/output/dir/output.rttm")
hypothesis = list(hypothesis.values())[0]  # Extract hypothesis from dictionary
reference = load_rttm("/path/to/reference.rttm")
reference = list(reference.values())[0]  # Extract reference from dictionary

der = metric(reference, hypothesis)
```

For convenience and to facilitate future comparisons, we also provide the [expected outputs](/expected_outputs) in RTTM format for every entry of Table 1 and Figure 5 in the paper. This includes the VBx offline topline as well as our proposed online approach with latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

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
