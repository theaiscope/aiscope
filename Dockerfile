FROM continuumio/anaconda3
ADD ./aiscope /opt/aiscope
WORKDIR /opt/aiscope
RUN pip install tensorflow
RUN pip install keras
RUN conda install -y -c menpo opencv
RUN mkdir /opt/aiscope/images
ADD mod-40_loss-2.0113.h5 /opt/aiscope
RUN apt-get install git -y
RUN pip install git+https://github.com/i008/keras-retinanet
