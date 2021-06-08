#!/bin/bash

# docker run -it --name tensorflow-gpu-jupyter tensorflow/tensorflow:latest-gpu-jupyter
# docker commit tensorflow-gpu-jupyter-tmp tensorflow-gpu-jupyter

docker run -it --rm --gpus all -p 8888:8888 -v "/home/fabian/Documents/studium/Imitation-Learning:/opt/project" \
  -e JUPYTER_TOKEN=b0355f51bc6f93f72553da74bb6548801e64b2f9689ad96c \
  tensorflow-gpu-jupyter