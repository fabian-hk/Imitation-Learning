FROM tensorflow/tensorflow:2.5.0-gpu-jupyter

COPY requirements.txt /requirements.txt

RUN apt update && apt install -y ffmpeg libsm6 libxext6

RUN python3 -m pip install --upgrade pip
RUN pip3 install -r /requirements.txt