# Docker file to build and test jet-engine-inspection
#
# configure /etc/default/docker
# DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4 --ip-masq=true"
FROM ubuntu:xenial
RUN apt-get update
RUN apt-get install -q -y build-essential
RUN apt-get install -q -y python
RUN apt-get install -q -y python-dev
RUN apt-get install -q -y python-pip
RUN apt-get install -q -y python-pytest
RUN apt-get install -q -y python-opencv

RUN mkdir -p /usr/src/machine-learning-tutorial
WORKDIR /usr/src/machine-learning-tutorial
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD machine-learning-tutorial.tar .
RUN pip freeze
