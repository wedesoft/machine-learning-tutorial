# Docker file to build and test jet-engine-inspection
#
# configure /etc/default/docker
# DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4 --ip-masq=true"
FROM ubuntu:xenial
RUN apt-get update
RUN apt-get install -q -y python
RUN apt-get install -q -y python-pytest
RUN apt-get install -q -y python-numpy
RUN apt-get install -q -y python-opencv

RUN mkdir -p /usr/src/machine-learning-tutorial
WORKDIR /usr/src/machine-learning-tutorial

ADD machine-learning-tutorial.tar .
