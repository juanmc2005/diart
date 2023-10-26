<br/>

<p align="center">
<img width="50%" src="/logo.jpg" title="Logo" />
</p>

<p align="center">
<img alt="PyPI Version" src="https://img.shields.io/pypi/v/diart?color=g">
<img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/diart?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads">
<img alt="Top language" src="https://img.shields.io/github/languages/top/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="Code size in bytes" src="https://img.shields.io/github/languages/code-size/juanmc2005/StreamingSpeakerDiarization?color=g">
<img alt="License" src="https://img.shields.io/github/license/juanmc2005/StreamingSpeakerDiarization?color=g">
<a href="https://joss.theoj.org/papers/cc9807c6de75ea4c29025c7bd0d31996"><img src="https://joss.theoj.org/papers/cc9807c6de75ea4c29025c7bd0d31996/status.svg"></a>
</p>

<div align="center">
  <h4>
    <a href="#-installation">
      💾 Installation
    </a>
    <span> | </span>
    <a href="#%EF%B8%8F-stream-audio">
      🎙️ Stream audio
    </a>
    <span> | </span>
    <a href="#-custom-models">
      🤖 Add your model
    </a>
    <span> | </span>
    <a href="#-tune-hyper-parameters">
      📈 Tune hyper-parameters
    </a>
    <br />
    <a href="#-build-pipelines">
      🧠🔗 Build pipelines
    </a>
    <span> | </span>
    <a href="#-websockets">
      🌐 WebSockets
    </a>
    <span> | </span>
    <a href="#-powered-by-research">
      🔬 Research
    </a>
    <span> | </span>
    <a href="#-citation">
      📗 Citation
    </a>
    <span> | </span>
    <a href="#-reproducibility">
      👨‍💻 Reproducibility
    </a>
  </h4>
</div>

<br/>

<p align="center">
<img width="100%" src="/demo.gif" title="Real-time diarization example" />
</p>

## 💾 Installation

1) Create environment:

```shell
conda env create -f diart/environment.yml
conda activate diart
```

2) Install the package:
```shell
pip install diart
```

### Get access to 🎹 pyannote models

By default, diart is based on [pyannote.audio](https://github.com/pyannote/pyannote-audio) models stored in the [huggingface](https://huggingface.co/) hub.
To allow diart to use them, you need to follow these steps:

1) [Accept user conditions](https://huggingface.co/pyannote/segmentation) for the `pyannote/segmentation` model
2) [Accept user conditions](https://huggingface.co/pyannote/embedding) for the `pyannote/embedding` model
3) Install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start#install-the-hub-library) and [log in](https://huggingface.co/docs/huggingface_hub/quick-start#login) with your user access token (or provide it manually in diart CLI or API).

## 🎙️ Stream audio

### From the command line

A recorded conversation:

```shell
diart.stream /path/to/audio.wav
```

A live conversation:

```shell
# Use "microphone:ID" to select a non-default device
# See `python -m sounddevice` for available devices
diart.stream microphone
```

See `diart.stream -h` for more options.

### From python

Use `StreamingInference` to run a pipeline on an audio source and write the results to disk:

```python
from diart import SpeakerDiarization
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference
from diart.sinks import RTTMWriter

pipeline = SpeakerDiarization()
mic = MicrophoneAudioSource()
inference = StreamingInference(pipeline, mic, do_plot=True)
inference.attach_observers(RTTMWriter(mic.uri, "/output/file.rttm"))
prediction = inference()
```

For inference and evaluation on a dataset we recommend to use `Benchmark` (see notes on [reproducibility](#-reproducibility)).

## 🤖 Add your model

Third-party models can be integrated by subclassing `SegmentationModel` and `EmbeddingModel` (both PyTorch `nn.Module`):

```python
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import EmbeddingModel, SegmentationModel
from diart.sources import MicrophoneAudioSource
from diart.inference import StreamingInference


def model_loader():
    return load_pretrained_model("my_model.ckpt")


class MySegmentationModel(SegmentationModel):
    def __init__(self):
        super().__init__(model_loader)
    
    @property
    def sample_rate(self) -> int:
        return 16000
    
    @property
    def duration(self) -> float:
        return 2  # seconds
    
    def forward(self, waveform):
        # self.model is created lazily
        return self.model(waveform)

    
class MyEmbeddingModel(EmbeddingModel):
    def __init__(self):
        super().__init__(model_loader)
    
    def forward(self, waveform, weights):
        # self.model is created lazily
        return self.model(waveform, weights)

    
config = SpeakerDiarizationConfig(
    segmentation=MySegmentationModel(),
    embedding=MyEmbeddingModel()
)
pipeline = SpeakerDiarization(config)
mic = MicrophoneAudioSource()
inference = StreamingInference(pipeline, mic)
prediction = inference()
```

## 📈 Tune hyper-parameters

Diart implements an optimizer based on [optuna](https://optuna.readthedocs.io/en/stable/index.html) that allows you to tune pipeline hyper-parameters to your needs.

### From the command line

```shell
diart.tune /wav/dir --reference /rttm/dir --output /output/dir
```

See `diart.tune -h` for more options.

### From python

```python
from diart.optim import Optimizer

optimizer = Optimizer("/wav/dir", "/rttm/dir", "/output/dir")
optimizer(num_iter=100)
```

This will write results to an sqlite database in `/output/dir`.

### Distributed optimization

For bigger datasets, it is sometimes more convenient to run multiple optimization processes in parallel.
To do this, create a study on a [recommended DBMS](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html#sphx-glr-tutorial-10-key-features-004-distributed-py) (e.g. MySQL or PostgreSQL) making sure that the study and database names match:

```shell
mysql -u root -e "CREATE DATABASE IF NOT EXISTS example"
optuna create-study --study-name "example" --storage "mysql://root@localhost/example"
```

You can now run multiple identical optimizers pointing to this database:

```shell
diart.tune /wav/dir --reference /rttm/dir --storage mysql://root@localhost/example
```

or in python:

```python
from diart.optim import Optimizer
from optuna.samplers import TPESampler
import optuna

db = "mysql://root@localhost/example"
study = optuna.load_study("example", db, TPESampler())
optimizer = Optimizer("/wav/dir", "/rttm/dir", study)
optimizer(num_iter=100)
```

## 🧠🔗 Build pipelines

For a more advanced usage, diart also provides building blocks that can be combined to create your own pipeline.
Streaming is powered by [RxPY](https://github.com/ReactiveX/RxPY), but the `blocks` module is completely independent and can be used separately.

### Example

Obtain overlap-aware speaker embeddings from a microphone stream:

```python
import rx.operators as ops
import diart.operators as dops
from diart.sources import MicrophoneAudioSource
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding

segmentation = SpeakerSegmentation.from_pyannote("pyannote/segmentation")
embedding = OverlapAwareSpeakerEmbedding.from_pyannote("pyannote/embedding")
mic = MicrophoneAudioSource()

stream = mic.stream.pipe(
    # Reformat stream to 5s duration and 500ms shift
    dops.rearrange_audio_stream(sample_rate=segmentation.model.sample_rate),
    ops.map(lambda wav: (wav, segmentation(wav))),
    ops.starmap(embedding)
).subscribe(on_next=lambda emb: print(emb.shape))

mic.read()
```

Output:

```
# Shape is (batch_size, num_speakers, embedding_dim)
torch.Size([1, 3, 512])
torch.Size([1, 3, 512])
torch.Size([1, 3, 512])
...
```

## 🌐 WebSockets

Diart is also compatible with the WebSocket protocol to serve pipelines on the web.

### From the command line

```commandline
diart.serve --host 0.0.0.0 --port 7007
diart.client microphone --host <server-address> --port 7007
```

**Note:** make sure that the client uses the same `step` and `sample_rate` than the server with `--step` and `-sr`.

See `-h` for more options.

### From python

For customized solutions, a server can also be created in python using the `WebSocketAudioSource`:

```python
from diart import SpeakerDiarization
from diart.sources import WebSocketAudioSource
from diart.inference import StreamingInference

pipeline = SpeakerDiarization()
source = WebSocketAudioSource(pipeline.config.sample_rate, "localhost", 7007)
inference = StreamingInference(pipeline, source)
inference.attach_hooks(lambda ann_wav: source.send(ann_wav[0].to_rttm()))
prediction = inference()
```

## 🔬 Powered by research

Diart is the official implementation of the paper *[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](/paper.pdf)* by [Juan Manuel Coria](https://juanmc2005.github.io/), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay](https://saharghannay.github.io/) and [Sophie Rosset](https://perso.limsi.fr/rosset/).


> We propose to address online speaker diarization as a combination of incremental clustering and local diarization applied to a rolling buffer updated every 500ms. Every single step of the proposed pipeline is designed to take full advantage of the strong ability of a recently proposed end-to-end overlap-aware segmentation to detect and separate overlapping speakers. In particular, we propose a modified version of the statistics pooling layer (initially introduced in the x-vector architecture) to give less weight to frames where the segmentation model predicts simultaneous speakers. Furthermore, we derive cannot-link constraints from the initial segmentation step to prevent two local speakers from being wrongfully merged during the incremental clustering step. Finally, we show how the latency of the proposed approach can be adjusted between 500ms and 5s to match the requirements of a particular use case, and we provide a systematic analysis of the influence of latency on the overall performance (on AMI, DIHARD and VoxConverse).

<p align="center">
<img height="400" src="/figure1.png" title="Visual explanation of the system" width="325" />
</p>

## 📗 Citation

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

## 👨‍💻 Reproducibility

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

`diart.benchmark` and `diart.inference.Benchmark` can run, evaluate and measure the real-time latency of the pipeline. For instance, for a DIHARD III configuration:

```shell
diart.benchmark /wav/dir --reference /rttm/dir --tau-active=0.555 --rho-update=0.422 --delta-new=1.517 --segmentation pyannote/segmentation@Interspeech2021
```

or using the inference API:

```python
from diart.inference import Benchmark, Parallelize
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import SegmentationModel

benchmark = Benchmark("/wav/dir", "/rttm/dir")

name = "pyannote/segmentation@Interspeech2021"
segmentation = SegmentationModel.from_pyannote(name)
config = SpeakerDiarizationConfig(
    # Set the model used in the paper
    segmentation=segmentation,
    step=0.5,
    latency=0.5,
    tau_active=0.555,
    rho_update=0.422,
    delta_new=1.517
)
benchmark(SpeakerDiarization, config)

# Run the same benchmark in parallel
p_benchmark = Parallelize(benchmark, num_workers=4)
if __name__ == "__main__":  # Needed for multiprocessing
    p_benchmark(SpeakerDiarization, config)
```

This pre-calculates model outputs in batches, so it runs a lot faster.
See `diart.benchmark -h` for more options.

For convenience and to facilitate future comparisons, we also provide the [expected outputs](/expected_outputs) of the paper implementation in RTTM format for every entry of Table 1 and Figure 5. This includes the VBx offline topline as well as our proposed online approach with latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

![Figure 5](/figure5.png)

## 📑 License

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
