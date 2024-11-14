# Use NVIDIA's CUDA base image
FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Author
LABEL author="Anjana Senanayake"

# Set a working directory
WORKDIR /workspace

# Set the environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# Install required packages and set up timezone
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Australia/Melbourne /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    git \
    libx11-6 \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_23.1.0-1-Linux-x86_64.sh \
 && chmod +x /tmp/miniconda.sh \
 && /tmp/miniconda.sh -b -p /opt/conda \
 && rm /tmp/miniconda.sh

# Add Conda to PATH
ENV PATH /opt/conda/bin:$PATH
ENV CUDA_HOME /usr/local/cuda
ENV MPLCONFIGDIR /tmp/matplotlib_config

# Install Python and packages
RUN conda install -y python=3.12
RUN conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
RUN conda clean -ya

# Install additional packages with pip
RUN pip install \
    argparse \
    matplotlib \
    ninja \
    opencv-python-headless \
    pandas \
    scikit-image \
    tensorboard \
    tqdm

CMD ["/bin/bash"]
