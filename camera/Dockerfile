# Use an Ubuntu 18.04 base image, as the original project was designed for it.
FROM ubuntu:18.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies, including Python 3.6, pip, and other libraries
# needed for opencv, dlib, and face_recognition.
RUN apt-get update && apt-get install -y \
    python3.6 \
    python3.6-dev \
    python3-pip \
    cmake \
    libboost-python-dev \
    libboost-thread-dev \
    libatlas-base-dev \
    libopenblas-dev \
    git \
    wget \
    && apt-get clean

# Update pip and set python3.6 as the default python3
RUN python3.6 -m pip install --upgrade pip && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1

# Copy the camera application code into the container
WORKDIR /app
COPY . .

# Install Python dependencies from requirements.txt
RUN pip3 install -r requirements.txt

# Install dlib and face_recognition (these compile from source and can take time)
RUN pip3 install dlib==19.21.1
RUN pip3 install face_recognition==1.3.0


# Install mediapipe for python 3.6 from a pre-built wheel
RUN wget https://github.com/PINTO0309/mediapipe-bin/releases/download/v0.8.5/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl
RUN pip3 install mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl
RUN rm mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64.whl


# Expose the port for the websocket bridge
EXPOSE 8082

# Set the default command to run the main application
CMD ["python3", "main.py"]
