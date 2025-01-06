<p align="center">
<img width="100%" src="https://github.com/juanmc2005/diart/blob/main/logo.jpg?raw=true" title="diart logo" />
</p>

<p align="center">
<i>🌿 Build AI-powered real-time audio applications in a breeze 🌿</i>
</p>

<p align="center">
<img alt="PyPI Version" src="https://img.shields.io/pypi/v/diart?color=g">
<img alt="PyPI Downloads" src="https://static.pepy.tech/personalized-badge/diart?period=total&units=international_system&left_color=grey&right_color=brightgreen&left_text=downloads">
<img alt="Python Versions" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-dark_green">
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
    <a href="#-models">
      🧠 Models
    </a>
    <br />
    <a href="#-tune-hyper-parameters">
      📈 Tuning
    </a>
    <span> | </span>
    <a href="#-build-pipelines">
      🧠🔗 Pipelines
    </a>
    <span> | </span>
    <a href="#-websockets">
      🌐 WebSockets
    </a>
    <span> | </span>
    <a href="#-powered-by-research">
      🔬 Research
    </a>
  </h4>
</div>

<br/>

<p align="center">
<img width="100%" src="https://github.com/juanmc2005/diart/blob/main/demo.gif?raw=true" title="Real-time diarization example" />
</p>

## ⚡ Quick introduction

Diart is a python framework to build AI-powered real-time audio applications.
Its key feature is the ability to recognize different speakers in real time with state-of-the-art performance,
a task commonly known as "speaker diarization".

The pipeline `diart.SpeakerDiarization` combines a speaker segmentation and a speaker embedding model
to power an incremental clustering algorithm that gets more accurate as the conversation progresses:

<p align="center">
<img width="100%" src="https://github.com/juanmc2005/diart/blob/main/pipeline.gif?raw=true" title="Real-time speaker diarization pipeline" />
</p>

With diart you can also create your own custom AI pipeline, benchmark it,
tune its hyper-parameters, and even serve it on the web using websockets.

**We provide pre-trained pipelines for:**

- Speaker Diarization
- Voice Activity Detection
- Transcription ([coming soon](https://github.com/juanmc2005/diart/pull/144))
- [Speaker-Aware Transcription](https://betterprogramming.pub/color-your-captions-streamlining-live-transcriptions-with-diart-and-openais-whisper-6203350234ef) ([coming soon](https://github.com/juanmc2005/diart/pull/147))

## 💾 Installation

**1) Make sure your system has the following dependencies:**

```
ffmpeg < 4.4
portaudio == 19.6.X
libsndfile >= 1.2.2
```

Alternatively, we provide an `environment.yml` file for a pre-configured conda environment:

```shell
conda env create -f diart/environment.yml
conda activate diart
```

**2) Install the package:**
```shell
pip install diart
```

### Get access to 🎹 pyannote models

By default, diart is based on [pyannote.audio](https://github.com/pyannote/pyannote-audio) models from the [huggingface](https://huggingface.co/) hub.
In order to use them, please follow these steps:

1) [Accept user conditions](https://huggingface.co/pyannote/segmentation) for the `pyannote/segmentation` model
2) [Accept user conditions](https://huggingface.co/pyannote/segmentation-3.0) for the newest `pyannote/segmentation-3.0` model
3) [Accept user conditions](https://huggingface.co/pyannote/embedding) for the `pyannote/embedding` model
4) Install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/quick-start#install-the-hub-library) and [log in](https://huggingface.co/docs/huggingface_hub/quick-start#login) with your user access token (or provide it manually in diart CLI or API).

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

By default, diart runs a speaker diarization pipeline, equivalent to setting `--pipeline SpeakerDiarization`,
but you can also set it to `--pipeline VoiceActivityDetection`. See `diart.stream -h` for more options.

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

For inference and evaluation on a dataset we recommend to use `Benchmark` (see notes on [reproducibility](#reproducibility)).

## 🧠 Models

You can use other models with the `--segmentation` and `--embedding` arguments.
Or in python:

```python
import diart.models as m

segmentation = m.SegmentationModel.from_pretrained("model_name")
embedding = m.EmbeddingModel.from_pretrained("model_name")
```

### Pre-trained models

Below is a list of all the models currently supported by diart:

| Model Name                                                                                                                | Model Type   | CPU Time* | GPU Time* |
|---------------------------------------------------------------------------------------------------------------------------|--------------|-----------|-----------|
| [🤗](https://huggingface.co/pyannote/segmentation) `pyannote/segmentation` (default)                                      | segmentation | 12ms      | 8ms       |
| [🤗](https://huggingface.co/pyannote/segmentation-3.0) `pyannote/segmentation-3.0`                                        | segmentation | 11ms      | 8ms       |
| [🤗](https://huggingface.co/pyannote/embedding) `pyannote/embedding` (default)                                            | embedding | 26ms      | 12ms      |
| [🤗](https://huggingface.co/hbredin/wespeaker-voxceleb-resnet34-LM) `hbredin/wespeaker-voxceleb-resnet34-LM` (ONNX)       | embedding | 48ms      | 15ms      |
| [🤗](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) `pyannote/wespeaker-voxceleb-resnet34-LM` (PyTorch)  | embedding | 150ms     | 29ms      |
| [🤗](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) `speechbrain/spkrec-xvect-voxceleb`                        | embedding | 41ms      | 15ms      |
| [🤗](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) `speechbrain/spkrec-ecapa-voxceleb`                        | embedding | 41ms      | 14ms      |
| [🤗](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb-mel-spec) `speechbrain/spkrec-ecapa-voxceleb-mel-spec`      | embedding | 42ms      | 14ms      |
| [🤗](https://huggingface.co/speechbrain/spkrec-resnet-voxceleb) `speechbrain/spkrec-resnet-voxceleb`                      | embedding | 41ms      | 16ms      |
| [🤗](https://huggingface.co/nvidia/speakerverification_en_titanet_large) `nvidia/speakerverification_en_titanet_large`    | embedding | 91ms      | 16ms      |

The latency of segmentation models is measured in a VAD pipeline (5s chunks).

The latency of embedding models is measured in a diarization pipeline using `pyannote/segmentation` (also 5s chunks).

\* CPU: AMD Ryzen 9 - GPU: RTX 4060 Max-Q

### Custom models

Third-party models can be integrated by providing a loader function:

```python
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.models import EmbeddingModel, SegmentationModel

def segmentation_loader():
    # It should take a waveform and return a segmentation tensor
    return load_pretrained_model("my_model.ckpt")

def embedding_loader():
    # It should take (waveform, weights) and return per-speaker embeddings
    return load_pretrained_model("my_other_model.ckpt")

segmentation = SegmentationModel(segmentation_loader)
embedding = EmbeddingModel(embedding_loader)
config = SpeakerDiarizationConfig(
    # Set the segmentation model used in the paper
    segmentation=segmentation,
    embedding=embedding,
)
pipeline = SpeakerDiarization(config)
```

If you have an ONNX model, you can use `from_onnx()`:

```python
from diart.models import EmbeddingModel

embedding = EmbeddingModel.from_onnx(
    model_path="my_model.ckpt",
    input_names=["x", "w"],  # defaults to ["waveform", "weights"]
    output_name="output",  # defaults to "embedding"
)
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

### Distributed tuning

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
from diart.sources import MicrophoneAudioSource, FileAudioSource
from diart.blocks import SpeakerSegmentation, OverlapAwareSpeakerEmbedding

segmentation = SpeakerSegmentation.from_pretrained("pyannote/segmentation")
embedding = OverlapAwareSpeakerEmbedding.from_pretrained("pyannote/embedding")

source = MicrophoneAudioSource()
# To take input from file:
# source = FileAudioSource("<filename>", sample_rate=16000)

# Make sure the models have been trained with this sample rate
print(source.sample_rate)

stream = mic.stream.pipe(
    # Reformat stream to 5s duration and 500ms shift
    dops.rearrange_audio_stream(sample_rate=source.sample_rate),
    ops.map(lambda wav: (wav, segmentation(wav))),
    ops.starmap(embedding)
).subscribe(on_next=lambda emb: print(emb.shape))

source.read()
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

```shell
diart.serve --host 0.0.0.0 --port 7007
diart.client microphone --host <server-address> --port 7007
```

**Note:** make sure that the client uses the same `step` and `sample_rate` than the server with `--step` and `-sr`.

See `-h` for more options.

### From the Dockerfile

You can also run the server in a Docker container. First, build the image:
```shell
docker build -t diart -f Dockerfile .
```

Run the server with default configuration:
```shell
docker run -p 7007:7007 --gpus all -e HF_TOKEN=<token> diart
```

Run with custom configuration:
```shell
docker run -p 7007:7007 --restart unless-stopped --gpus all \
  -e HF_TOKEN=<token> \
  -e HOST=0.0.0.0 \
  -e PORT=7007 \
  -e SEGMENTATION=pyannote/segmentation-3.0 \
  -e EMBEDDING=speechbrain/spkrec-resnet-voxceleb \
  -e TAU_ACTIVE=0.45 \
  -e RHO_UPDATE=0.25 \
  -e DELTA_NEW=0.6 \
  -e LATENCY=5 \
  -e MAX_SPEAKERS=3 \
  diart
```

The server can be configured using these environment variables, at runtime:
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 7007)
- `SEGMENTATION`: Segmentation model (default: pyannote/segmentation)
- `EMBEDDING`: Embedding model (default: pyannote/embedding)
- `TAU_ACTIVE`: Activity threshold (default: 0.5)
- `RHO_UPDATE`: Update threshold (default: 0.3)
- `DELTA_NEW`: New speaker threshold (default: 1.0)
- `LATENCY`: Processing latency in seconds (default: 0.5)
- `MAX_SPEAKERS`: Maximum number of speakers (default: 20)

### From python

For customized solutions, a server can also be created in python using `WebSocketStreamingServer`:

```python
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.websockets import WebSocketStreamingServer

pipeline_class = SpeakerDiarization
pipeline_config = SpeakerDiarizationConfig(step=0.5, sample_rate=16000)
server = WebSocketStreamingServer(pipeline_class, pipeline_config, host="localhost", port=7007)
server.run()
```

## 🔬 Powered by research

Diart is the official implementation of the paper
[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](https://github.com/juanmc2005/diart/blob/main/paper.pdf)
by [Juan Manuel Coria](https://juanmc2005.github.io/),
[Hervé Bredin](https://herve.niderb.fr),
[Sahar Ghannay](https://saharghannay.github.io/)
and [Sophie Rosset](https://perso.limsi.fr/rosset/).


> We propose to address online speaker diarization as a combination of incremental clustering and local diarization applied to a rolling buffer updated every 500ms. Every single step of the proposed pipeline is designed to take full advantage of the strong ability of a recently proposed end-to-end overlap-aware segmentation to detect and separate overlapping speakers. In particular, we propose a modified version of the statistics pooling layer (initially introduced in the x-vector architecture) to give less weight to frames where the segmentation model predicts simultaneous speakers. Furthermore, we derive cannot-link constraints from the initial segmentation step to prevent two local speakers from being wrongfully merged during the incremental clustering step. Finally, we show how the latency of the proposed approach can be adjusted between 500ms and 5s to match the requirements of a particular use case, and we provide a systematic analysis of the influence of latency on the overall performance (on AMI, DIHARD and VoxConverse).

<p align="center">
<img height="400" src="https://github.com/juanmc2005/diart/blob/main/figure1.png?raw=true" title="Visual explanation of the system" width="325" />
</p>

### Citation

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

### Reproducibility

![Results table](https://github.com/juanmc2005/diart/blob/main/table1.png?raw=true)

**Important:** We highly recommend installing `pyannote.audio<3.1` to reproduce these results.
For more information, see [this issue](https://github.com/juanmc2005/diart/issues/214).

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

model_name = "pyannote/segmentation@Interspeech2021"
model = SegmentationModel.from_pretrained(model_name)
config = SpeakerDiarizationConfig(
    # Set the segmentation model used in the paper
    segmentation=model,
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

For convenience and to facilitate future comparisons, we also provide the
<a href="https://github.com/juanmc2005/diart/tree/main/expected_outputs">expected outputs</a>
of the paper implementation in RTTM format for every entry of Table 1 and Figure 5.
This includes the VBx offline topline as well as our proposed online approach with
latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

![Figure 5](https://github.com/juanmc2005/diart/blob/main/figure5.png?raw=true)

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

<p style="color:grey;font-size:14px;">Logo generated by <a href="https://www.designevo.com/" title="Free Online Logo Maker">DesignEvo free logo designer</a></p>
