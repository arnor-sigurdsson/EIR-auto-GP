FROM nvcr.io/nvidia/pytorch:22.05-py3

LABEL maintainer="Arnor Ingi Sigurdsson" \
      version="0.1" \
      description="This Docker image contains EIR-auto-GP and its dependencies." \
      license="APGL"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv

ENV VIRTUAL_ENV=/opt/venv
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3.10 -m pip install eir-auto-gp

RUN apt-get install -y curl unzip
RUN curl -LJO https://s3.amazonaws.com/plink2-assets/alpha3/plink2_linux_avx2_20221024.zip && \
    unzip plink2_*.zip && \
    rm plink2_*.zip && \
    chmod +x plink2 && \
    mv plink2 /usr/local/bin && \
    plink2 --version


RUN apt-get clean

CMD ["/bin/bash"]

