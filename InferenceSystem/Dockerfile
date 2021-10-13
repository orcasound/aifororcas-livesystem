# Specify base image
FROM continuumio/miniconda3

# Install sudo apt-get requirements
RUN apt-get update && apt-get install -y \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libglib2.0-0 \
        ffmpeg \
        unzip \
        git

# Install packages
RUN  conda install -c pytorch -c fastai fastai=1.0.61 -y
RUN pip install numba==0.42.0
RUN pip install git+https://github.com/fastaudio/fastai_audio@0.1
RUN conda install -c conda-forge librosa pydub -y
RUN pip install ipython

# Install pip requirements
WORKDIR /usr/src/install
COPY ./requirements.txt ./requirements.txt
RUN python3 -m pip install --upgrade pip && pip install -r requirements.txt

# From now on, we will be working in the app's running directory
WORKDIR /usr/src/app

# Put model in place at WORKDIR/model
COPY ./model.zip ./model.zip
RUN unzip ./model.zip
RUN rm ./model.zip

# Move source code to app working directory
COPY ./src/ ./src
COPY ./config/ ./config

CMD ["python3", "-u", "./src/LiveInferenceOrchestrator.py", "--config", "./config/Production/FastAI_LiveHLS_OrcasoundLab.yml"]
