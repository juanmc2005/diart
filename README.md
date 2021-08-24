# Streaming speaker diarization

This is the companion repository for paper 

*[Overlap-aware low-latency online speaker diarization based on end-to-end local segmentation](/paper.pdf)*, by 
[Juan Manuel Coria](), [Hervé Bredin](https://herve.niderb.fr), [Sahar Ghannay]() and [Sophie Rosset]().

insert PNG version of Figure 1 here 

## Citation

```bibtex

```


## Installation

...

## Usage

### CLI

python from_file
python from_microphone

### API

```python
from ... import ...

```

##  Reproducible research

insert PNG version of Table 1 here

In order to reproduce the results of the paper, use the following hyper-parameters:

Dataset     | latency | $\tau$ | $\rho$ | $\delta$ 
------------|---------|--------|--------|----------
DIHARD III  | any     |        |        |   
AMI         | any     |        |        |   
VoxConverse | any     |        |        |   
DIHARD II   | 1s      |        |        |   
DIHARD II   | 5s      |        |        |

For instance, for DIHARD III configuration, one would use 

```python
from ... import OnlineSpeakerDiarization
pipeline = OnlineSpeakerDiarization(latency=5.0, tau=..., rho=..., delta=...)
result = pipeline("/path/to/audio.wav")

from pyannote.metrics.diarization import DiarizationErrorRate
metric = DiarizationErrorRate()
der = metric(reference, hypothesis)
```

For convenience and to facilitate future comparisons, we also provide the [expected outputs](/expected_outputs) in RTTM format corresponding to every entry of Table 1 and Figure 5 in the paper.  

This includes the VBx offline baseline as well as our proposed online approach with latencies 500ms, 1s, 2s, 3s, 4s, and 5s.

insert PNG version of Figure 5 here

