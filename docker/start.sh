#!/bin/bash

docker build -t tensorflow250-gpu-jupyter -f docker/Dockerfile .

docker run -it --rm --name jupyter --gpus all \
  -p 8888:8888 \
  -v "$PWD:/opt/project" \
  -e JUPYTER_TOKEN=b0355f51bc6f93f72553da74bb6548801e64b2f9689ad96c \
  -e MPLCONFIGDIR=/tmp \
  --user 1000:1000 \
  tensorflow250-gpu-jupyter jupyter notebook --no-browser --ip=0.0.0.0 --notebook-dir=/opt/project
