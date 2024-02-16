FROM python:3.7.11-slim-buster

# Set up the development environment
WORKDIR /Jingle

# Install extra packages
RUN apt update && \
    apt install nano

# Install dependencies and numpy (for dragonfly)
COPY requirements.txt setup.py ./
RUN pip install -U pip && \
    pip install --no-cache-dir numpy==1.21.2 && \
    pip install --no-cache-dir -r requirements.txt

# Expose ports for Scheduler (grpc)
ENV PORT 10000
EXPOSE $PORT

# Copy the significant files
COPY worker ./worker/
COPY utility ./utility/
COPY scheduler ./scheduler/
#COPY experiments ./experiments/
COPY real_experiments ./real_experiments/

# Install Jingle project
RUN pip install -e .

ENV PYTHONUNBUFFERED 1

ENV PYTHONPATH "${PYTHONPATH}:/"

ENV Jingle_SERVICE_SERVICE_PORT 10000
#ENV Jingle_SERVICE_SERVICE_HOST dns:///127.0.0.1


#CMD ["python", "/Jingle/scheduler/driver/cluster_driver.py"]
CMD ["python", "/Jingle/scheduler/driver/cluster_driver.py"]