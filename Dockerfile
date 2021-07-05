FROM continuumio/miniconda3

ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/code

RUN python -m pip install --upgrade pip

RUN conda update -n base -c defaults conda

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "DeepPurpose4", "/bin/bash", "-c"]

COPY . .