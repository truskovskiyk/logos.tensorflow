FROM tensorflow/tensorflow:1.8.0-devel-py3
# basic setup
RUN apt-get update
RUN apt-get -y install protobuf-compiler python-pil python-lxml python-tk python3-tk

# install protobuf 3 on Ubuntu
RUN curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
RUN unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
RUN mv protoc3/bin/* /usr/local/bin/
RUN mv protoc3/include/* /usr/local/include/

# app setup
WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt


# install tensorflow models
RUN git clone https://github.com/truskovskiyk/models.git
RUN cd /app/models/research/ && protoc object_detection/protos/*.proto --python_out=.
RUN cp -r /app/models/research/object_detection /app/object_detection
RUN cp -r /app/models/research/slim /app/slim
ENV PYTHONPATH $PYTHONPATH:/app/:/app/slim


