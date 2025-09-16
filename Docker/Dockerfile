FROM python:3.8

ENV DEBIAN_FRONTEND=noninteractive
EXPOSE 22

RUN apt-get update &&  apt-get install -y \
    pip \
    --upgrade pip \
    curl \
    dialog apt-utils \
    ca-certificates \
    vim \
    nano \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    openssh-server \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
 && rm -rf /var/lib/apt/lists/*

#### pymesh section
#git clone https://github.com/PyMesh/PyMesh.git
#cd PyMesh
#git submodule update --init
#export PYMESH_PATH=`pwd`
#RUN apt-get install -y \
#    libeigen3-dev \
#    libgmp-dev \
#    libgmpxx4ldbl \
#    libmpfr-dev \
#    libboost-dev \
#    libboost-thread-dev \
#    libtbb-dev \
#    python3-dev

# follow up with requirements.txt


# create a non-root user
ARG USER_ID=1000
#ARG USER_PW="lastlogin"
RUN useradd -m --no-log-init --system  --uid   ${USER_ID} appuser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser


# make ssh stuff
ARG SSH_KEY
RUN sudo mkdir -p /var/run/sshd
RUN sudo mkdir -p /home/appuser/.ssh
ADD id_rsa.pub /home/appuser/
RUN sudo cp /home/appuser/id_rsa.pub /home/appuser/.ssh/authorized_keys
RUN sudo chown -R appuser:sudo /home/appuser


# create project dir
RUN sudo mkdir /home/appuser/scanplan
RUN sudo mkdir /home/appuser/scanplan/result_mesh


# setup
COPY requirements.txt .
ENV PATH "${PATH}:/home/appuser/.local/bin"
RUN pip install -r requirements.txt
RUN pip install deap
RUN pip install plotly
RUN pip install matplotlib-label-lines


#extra, relocate
RUN sudo chown -R appuser:sudo /home/appuser/scanplan