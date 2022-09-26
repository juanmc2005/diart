---
title: 'Diart: A Python Library for Real-Time Speaker Diarization'
tags:
  - Python
  - machine learning
  - artificial intelligence
  - speaker diarization
  - speaker embedding
  - speaker clustering
  - real time
  - streaming
authors:
  - name: Juan Manuel Coria
    orcid: 0000-0002-5035-147X
    corresponding: true
    affiliation: 1
  - name: Hervé Bredin
    orcid: 0000-0002-3739-925X
    affiliation: 2
  - name: Sahar Ghannay
    orcid: 0000-0002-7531-2522
    affiliation: 1 
  - name: Sophie Rosset
    orcid: 0000-0002-6865-4989
    affiliation: 1
affiliations:
  - name: Université Paris-Saclay CNRS, LISN, Orsay, France
    index: 1
  - name: IRIT, Université de Toulouse, CNRS, Toulouse, France
    index: 2
date: 23 September 2022
bibliography: paper.bib
---

# Summary

The term "speaker diarization" denotes the problem of determining
"who speaks when" in a recorded conversation. Among other reasons, it
has attracted the attention of the speech research community because of
its ability to improve transcription performance, readability and
exploitability. Speaker diarization in real-time holds the potential to
accelerate and cement the adoption of this technology in our everyday lives.
However, although "offline" systems today achieve outstanding performance
in pre-recorded conversations, additional problems of "online" real-time
diarization, like limited context and low latency, require flexible and
efficient solutions enabling both research and production-ready applications.

# Statement of need

`Diart` is a Python library for real-time speaker diarization. It leverages
data structures and pre-trained models available in `pyannote.audio`
[@pyannote.audio] to implement production-ready real-time inference on a variety
of audio streams like local and remote audio/video files, microphones, and even
WebSockets. Moreover, `Diart` was designed to facilitate research by providing
fast batched inference and hyper-parameter tuning thanks to and in full
compatibility with `Optuna` [@optuna].

`Diart` was designed with an object-oriented API fully capable of extension and
customization. Streaming is powered internally by ReactiveX extensions, but
available "blocks" allow users to mix and match different operations with any
streaming library they choose. A prototyping tool with a CLI is also provided to
quickly evaluate, profile, visualize and optimize custom systems.

`Diart` is based on previous research on low-latency online speaker diarization
[@Coria:2021] and allows to reproduce its results. It has also participated in the
recent Ego4D Audio-only Diarization Challenge [@Ego4D:2022], outperforming the
offline baseline by a large margin. We hope `Diart`'s flexibility, efficiency and
customization will allow for exciting new research and applications in online
speaker diarization.

# Acknowledgements

This work has been funded by Université Paris-Saclay under PhD contract number 2019-089.

# References