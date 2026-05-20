FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Non-interactive apt
ENV DEBIAN_FRONTEND=noninteractive

# Don't write .pyc files (avoids stale bytecode in mounted dirs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Required by Luu's fork — bypasses old 'sklearn' package alias check
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# CUDA device ordering by PCI bus (so GPU indices match nvidia-smi on host)
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# -----------------------------------------------------------------------
# System packages — Python 3.11 (via deadsnakes PPA, same as gbm-seg)
# -----------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3-pip \
    git \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Make python3.11 the default `python` and `python3`
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip to a version that handles modern wheels
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# -----------------------------------------------------------------------
# Python dependencies — only what's on top of nnUNetv2 fork
# (nnUNetv2 fork itself pulls torch, numpy, scipy, etc. via its setup.py)
# -----------------------------------------------------------------------
COPY ms-seg/requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

# -----------------------------------------------------------------------
# Common shared code (ServiceBase, GPUMonitor, contracts)
# -----------------------------------------------------------------------
COPY common/ ./common/

# -----------------------------------------------------------------------
# Luu's SmallLesionMRI fork — its slsseg/setup.py registers as 'nnunetv2'
# This REPLACES the stock nnunetv2 inside this container.
# -----------------------------------------------------------------------
COPY ms-seg/SmallLesionMRI/ ./SmallLesionMRI/
RUN cd SmallLesionMRI/slsseg && pip install --ignore-installed .

# The fork's setup.py pins numpy==1.26.4 and scipy==1.11.4 — these are
# pre-numpy-2.x constraints that are incompatible with the ABI of more
# recent scikit-image/batchgenerators wheels.  Override them to match
# the stack used by gbm-seg, so both services have ABI-compatible
# numpy/scipy binaries.
RUN pip install --no-cache-dir --force-reinstall \
    "numpy>=2.3,<3" \
    "scipy>=1.13"

# -----------------------------------------------------------------------
# Service code
# -----------------------------------------------------------------------
COPY ms-seg/src/ ./src/
COPY ms-seg/server_config.yaml ./server_config.yaml

# Manifest is read by ServiceBase at startup — kept one level up from src/
COPY ms-seg/manifest.yaml ./manifest.yaml

# Marker file for traceability — which commit of the fork is installed
COPY ms-seg/SMALLLESIONMRI_COMMIT.txt ./SMALLLESIONMRI_COMMIT.txt

# Staging dir for nnUNet-convention temporary files during inference
RUN mkdir -p /app/data/tmp

ENV PYTHONPATH=/app:$PYTHONPATH

EXPOSE 5001

CMD ["python", "src/service_server.py"]