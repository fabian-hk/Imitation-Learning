#!/bin/bash

# docker run -it --name tensorflow-gpu-jupyter tensorflow/tensorflow:latest-gpu-jupyter
# docker commit tensorflow-gpu-jupyter-tmp tensorflow-gpu-jupyter

docker run -it --rm --name jupyter --gpus all \
  -p 8888:8888 \
  -v "/home/fabian/Documents/studium/Imitation-Learning:/opt/project" \
  -v "/media/fabian/Disk1/Data/commaai:/opt/project/data/commaai" \
  -e JUPYTER_TOKEN=b0355f51bc6f93f72553da74bb6548801e64b2f9689ad96c \
  -e MPLCONFIGDIR=/tmp \
  --user 1000:1000 \
  tensorflow-gpu-jupyter jupyter notebook --no-browser --ip=0.0.0.0 --notebook-dir=/opt/project