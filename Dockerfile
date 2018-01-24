# Docker file to build and test jet-engine-inspection
#
# configure /etc/default/docker
# DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4 --ip-masq=true"
FROM ubuntu:artful
RUN sed -i 's/archive.ubuntu.com/unix-updates.comm.ad.roke.co.uk/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get install -q -y build-essential
RUN apt-get install -q -y texlive-bibtex-extra
RUN apt-get install -q -y texlive-xetex
RUN apt-get install -q -y inkscape
RUN apt-get install -q -y biber
RUN apt-get install -q -y python3
RUN apt-get install -q -y python3-dev
RUN apt-get install -q -y python3-pip
RUN apt-get install -q -y python3-pytest
RUN apt-get install -q -y python3-matplotlib

# Install OpenCV dependencies
RUN apt-get install -q -y cmake
RUN apt-get install -q -y libjpeg8-dev libtiff5-dev libpng-dev
RUN apt-get install -q -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get install -q -y libxvidcore-dev libx264-dev
RUN apt-get install -q -y libgtk-3-dev
RUN apt-get install -q -y libatlas-base-dev gfortran

# Build and install OpenCV
# http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/
WORKDIR /usr/src
RUN curl -s https://codeload.github.com/opencv/opencv/tar.gz/3.2.0 --output opencv-3.2.0.tar.gz
RUN tar xf opencv-3.2.0.tar.gz
WORKDIR /usr/src/opencv-3.2.0
# Add "-D WITH_IPP=OFF" if 3rd party IPP download fails.
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D PYTHON_EXECUTABLE=/usr/bin/python3 \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          .
RUN make -j `nproc`
RUN make install
RUN ldconfig

# Build and test
RUN mkdir -p /usr/local/src/machine-learning-tutorial
WORKDIR /usr/local/src/machine-learning-tutorial
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
ADD machine-learning-tutorial.tar .
