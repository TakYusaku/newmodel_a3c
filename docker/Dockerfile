FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt install -y python3.7
# RUN apt install -y python2.7
RUN apt update -y
RUN apt upgrade -y
RUN apt install -y python3-pip
RUN apt install -y python-pip
RUN apt install -y nano
RUN apt install -y git

RUN apt clean -y
RUN apt autoremove -y
RUN apt update -y
RUN apt upgrade -y
RUN apt autoremove -y
RUN apt autoclean -y

RUN pip3 install python-dateutil
RUN pip3 install joblib
RUN pip3 install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install gym
RUN pip3 install torchsummary
RUN pip3 install matplotlib
RUN pip3 install dataclasses
RUN pip install PyOpenGL PyOpenGL_accelerate
RUN pip3 install PyOpenGL PyOpenGL_accelerate

# mount volume
RUN mkdir /home/develop
VOLUME /home/develop

EXPOSE 9000

CMD echo 'successful'