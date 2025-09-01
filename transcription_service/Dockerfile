FROM mambaorg/micromamba:focal as base

USER root
WORKDIR /home/ts
SHELL ["/bin/bash", "-c"]

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN apt-get update && \
    apt-get install --no-install-recommends ca-certificates wget bash bzip2 git unzip -y && \
    apt-get clean autoremove --yes && \
    rm -rf /var/lib/{apt,dpkg,cache,log}


FROM base as dependencies
# Copy and install dependencies
COPY --chown=root:root transcription_service transcription_service/

RUN cd transcription_service/src/external && \
    git clone https://github.com/adefossez/demucs.git

RUN micromamba create -n ts python=3.9 -c conda-forge -y && \
    micromamba run -n ts pip install -e transcription_service && \
    micromamba run -n ts pip install -e transcription_service/src/external/demucs && \
    micromamba run -n ts pip uninstall -y PySoundFile && \
    micromamba run -n ts pip install PySoundFile && \
    micromamba clean --all -y


FROM mambaorg/micromamba:focal as runtime

USER root
# Packages required for opencv
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libgl1  \
    && rm -rf /var/lib/apt/lists/*

COPY --from=dependencies /home/ts/ /home/ts/
COPY --from=dependencies /opt/conda/envs/ts /opt/conda/envs/ts
WORKDIR /home/ts

FROM runtime as transcription_service
ENV PYTHONPATH="/home/ts/transcription_service"
ENTRYPOINT ["/opt/conda/envs/ts/bin/python", "transcription_service/src/main.py"]
