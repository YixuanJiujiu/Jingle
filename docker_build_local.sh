#!/usr/bin/env bash
set -e

docker build . -t yixuannine/jinglescheduler:latest
docker login
docker push yixuannine/jinglescheduler:latest
#kind load docker-image yixuannine/Jinglescheduler:latest
