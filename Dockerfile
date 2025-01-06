# Use NVIDIA CUDA base image
FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Install sudo, git, wget, gcc, g++, and other essential build tools
RUN apt-get update && \
    apt-get install -y sudo git wget build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Install Python 3.10 using Conda
RUN conda install python=3.10

# Upgrade pip and setuptools to avoid deprecation warnings
RUN pip install --upgrade pip setuptools

# Set Python 3.11 as default by creating a symbolic link
RUN ln -sf /opt/conda/bin/python3.10 /opt/conda/bin/python && \
    ln -sf /opt/conda/bin/python3.10 /usr/bin/python

# Verify installations
RUN python --version && \
    gcc --version && \
    g++ --version && \
    pip --version && \
    conda --version

# Create app directory and copy files
WORKDIR /diart
COPY . .

# Install diart dependencies
RUN conda install portaudio pysoundfile ffmpeg -c conda-forge
RUN pip install -e .

# Expose the port the app runs on
EXPOSE 7007

# Define environment variable to prevent Python from buffering stdout/stderr
# and writing byte code to file
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Define custom options as env variables with defaults
ENV HOST=0.0.0.0
ENV PORT=7007
ENV SEGMENTATION=pyannote/segmentation-3.0
ENV EMBEDDING=speechbrain/spkrec-resnet-voxceleb
ENV TAU_ACTIVE=0.45
ENV RHO_UPDATE=0.25
ENV DELTA_NEW=0.6
ENV LATENCY=5
ENV MAX_SPEAKERS=3

CMD ["sh", "-c", "python -m diart.console.serve --host ${HOST} --port ${PORT} --segmentation ${SEGMENTATION} --embedding ${EMBEDDING} --tau-active ${TAU_ACTIVE} --rho-update ${RHO_UPDATE} --delta-new ${DELTA_NEW} --latency ${LATENCY} --max-speakers ${MAX_SPEAKERS}"]

# Example run command with environment variables:
# docker run -p 7007:7007 --restart unless-stopped --gpus all \
#   -e HF_TOKEN=<token> \
#   -e HOST=0.0.0.0 \
#   -e PORT=7007 \
#   -e SEGMENTATION=pyannote/segmentation-3.0 \
#   -e EMBEDDING=speechbrain/spkrec-resnet-voxceleb \
#   -e TAU_ACTIVE=0.45 \
#   -e RHO_UPDATE=0.25 \
#   -e DELTA_NEW=0.6 \
#   -e LATENCY=5 \
#   -e MAX_SPEAKERS=3 \
#   diart-image