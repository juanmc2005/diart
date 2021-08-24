# Streaming speaker diarization

*[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](/paper.pdf)*   
by [Juan Manuel Coria](https://juanmc2005.github.io/), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay](https://saharghannay.github.io/) and [Sophie Rosset](https://perso.limsi.fr/rosset/).


> We propose to address online speaker diarization as a combination of incremental clustering and local diarization applied to a rolling buffer updated every 500ms. Every single step of the proposed pipeline is designed to take full advantage of the strong ability of a recently proposed end-to-end overlap-aware segmentation to detect and separate overlapping speakers. In particular, we propose a modified version of the statistics pooling layer (initially introduced in the x-vector architecture) to give less weight to frames where the segmentation model predicts simultaneous speakers. Furthermore, we derive cannot-link constraints from the initial segmentation step to prevent two local speakers from being wrongfully merged during the incremental clustering step. Finally, we show how the latency of the proposed approach can be adjusted between 500ms and 5s to match the requirements of a particular use case, and we provide a systematic analysis of the influence of latency on the overall performance (on AMI, DIHARD and VoxConverse).

<p align="center">
<img height="400" src="/figure1.png" title="Figure 1" width="325" />
</p>

## Citation

```bibtex
Paper currently under review.
```

## Installation

1) Create environment:

```shell
conda create -n diarization python==3.8
conda activate diarization
```

2) Install the latest PyTorch version following the [official instructions](https://pytorch.org/get-started/locally/#start-locally)

3) Install dependencies:
```shell
pip install -r requirements.txt
```

## Usage

### CLI

Stream a previously recorded conversation:

```shell
python main.py /path/to/audio.wav
```

Or use a real audio stream from your microphone:

```shell
python main.py microphone
```

This will launch a real-time visualization of the diarization outputs as they are produced by the system:

![Example of a state of the real-time output plot](/visualization.png)

By default, the script uses step = latency = 500ms, and it sets reasonable values for all hyper-parameters.
See `python main.py -h` for more information.

### API

We provide various building blocks that can be combined to process an audio stream.
Our streaming implementation is based on [RxPY](https://github.com/ReactiveX/RxPY), but the `functional` module is completely independent.

In this example we show how to obtain speaker embeddings from a microphone stream with Equation 2:

```python
from sources import MicrophoneAudioSource
from functional import FrameWiseModel, ChunkWiseModel, OverlappedSpeechPenalty, EmbeddingNormalization

mic = MicrophoneAudioSource(sample_rate=16000)

# Initialize independent modules
segmentation = FrameWiseModel("pyannote/segmentation")
embedding = ChunkWiseModel("pyannote/embedding")
osp = OverlappedSpeechPenalty(gamma=3, beta=10)
normalization = EmbeddingNormalization(norm=1)

# Branch the microphone stream to calculate segmentation
segmentation_stream = mic.stream.pipe(ops.map(segmentation))
# Join audio and segmentation stream to calculate speaker embeddings
embedding_stream = rx.zip(mic.stream, segmentation_stream).pipe(
    ops.starmap(lambda wave, seg: (wave, osp(seg))),
    ops.starmap(embedding),
    ops.map(normalization)
)

embedding_stream.suscribe(on_next=lambda emb: print(emb.shape))

mic.read()
```

Output:

```
(4, 512)
(4, 512)
(4, 512)
...
```

##  Reproducible research

![Table 1](/table1.png)

In order to reproduce the results of the paper, use the following hyper-parameters:

Dataset     | latency | tau    | rho    | delta 
------------|---------|--------|--------|------
DIHARD III  | any     | 0.555  | 0.422  | 1.517  
AMI         | any     | 0.507  | 0.006  | 1.057  
VoxConverse | any     | 0.576  | 0.915  | 0.648  
DIHARD II   | 1s      | 0.619  | 0.326  | 0.997  
DIHARD II   | 5s      | 0.555  | 0.422  | 1.517  

For instance, for a DIHARD III configuration, one would use:

```shell
python main.py /path/to/file.wav --latency=5 --tau=0.555 --rho=0.422 --delta=1.517 --output /output/dir
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

For convenience and to facilitate future comparisons, we also provide the [expected outputs](/expected_outputs) in RTTM format corresponding to every entry of Table 1 and Figure 5 in the paper. This includes the VBx offline baseline as well as our proposed online approach with latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

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

