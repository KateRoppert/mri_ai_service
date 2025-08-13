# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV FSLDIR=/usr/local/fsl
ENV PATH="${FSLDIR}/bin:${PATH}"
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLTCLSH="${FSLDIR}/bin/fsltclsh"
ENV FSLWISH="${FSLDIR}/bin/fslwish"
ENV ANTSPATH=/opt/ANTs/bin/
ENV PATH="${ANTSPATH}:${PATH}"
ENV FREESURFER_HOME=/opt/freesurfer
ENV SUBJECTS_DIR="${FREESURFER_HOME}/subjects"
ENV PATH="${FREESURFER_HOME}/bin:${PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    pkg-config \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    libglib2.0-0 \
    libglu1-mesa \
    libgomp1 \
    libquadmath0 \
    bc \
    dc \
    file \
    libfontconfig1 \
    libfreetype6 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libice6 \
    libxcursor1 \
    libxft2 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxt6 \
    libopenblas-base \
    sudo \
    libsm6 \
    libxext6 \
    tcsh \
    unzip \
    vim \
    dicom3tools \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install FSL - using correct syntax
# The -s flag skips shell configuration which is what we want in Docker
RUN cd /tmp && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    python3 fslinstaller.py -d ${FSLDIR} -s && \
    rm fslinstaller.py

# # Source FSL configuration
RUN echo "source ${FSLDIR}/etc/fslconf/fsl.sh" >> /etc/bash.bashrc

# # Install ANTs (uncomment when ready)
RUN mkdir -p /opt/ANTs && \
    cd /tmp && \
    wget https://github.com/ANTsX/ANTs/releases/download/v2.4.4/ants-2.4.4-ubuntu-22.04-X64-gcc.zip && \
    unzip ants-2.4.4-ubuntu-22.04-X64-gcc.zip -d /opt/ANTs && \
    rm ants-2.4.4-ubuntu-22.04-X64-gcc.zip

# # Install Freesurfer (uncomment when ready)
# RUN cd /tmp && \
#     wget https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz && \
#     tar -xzvf freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz && \
#     mv freesurfer /opt/ && \
#     rm -f freesurfer-linux-ubuntu22_amd64-7.3.2.tar.gz

# Create app directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with better error handling
# RUN pip3 install --no-cache-dir --upgrade pip && \
#     pip3 install --no-cache-dir -r requirements.txt


RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Install additional neuroimaging Python packages
# RUN pip3 install --no-cache-dir \
#     nibabel \
#     nilearn \
#     nipype \
#     dipy \
#     pydicom \
#     SimpleITK \
#     flask \
#     flask-cors

# Copy application files
COPY config/ ./config/
COPY mni_templates/ ./mni_templates/
COPY pipeline/ ./pipeline/
COPY scripts/ ./scripts/
COPY webapp/ ./webapp/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/output /app/tests

# Set permissions
RUN chmod -R 755 /app

# Expose Flask port
EXPOSE 5001

# # Copy the init script
# COPY scripts/docker_init.py /app/scripts/

# # Make sure the script is executable  
# RUN chmod +x /app/scripts/docker_init.py

# COPY scripts/docker_entrypoint.sh /app/scripts/
# RUN chmod +x /app/scripts/docker_entrypoint.sh

# Set the entrypoint to source FSL and run the app
# Ensure FreeSurfer is also sourced
#ENTRYPOINT ["/bin/bash", "-c", "source ${FSLDIR}/etc/fslconf/fsl.sh && source ${FREESURFER_HOME}/SetUpFreeSurfer.sh && cd /app && python3 webapp/app.py"]
#ENTRYPOINT ["python3 webapp/app.py"]
ENTRYPOINT ["/bin/bash", "-c", "source ${FSLDIR}/etc/fslconf/fsl.sh && cd /app && python3 webapp/app.py"]