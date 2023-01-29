# Starting point
FROM ubuntu:focal

# Get python installation ready
RUN apt-get update
RUN apt-get install -y gcc python3-dev python3-venv csh
# Create virtual environment
RUN python3 -m venv /usr/ve
# Copy dependencies and install
RUN  . /usr/ve/bin/activate && pip install --upgrade pip

COPY docker/py_packages.txt .
RUN . /usr/ve/bin/activate && pip install -r py_packages.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Install Java
RUN apt-get update
RUN apt-get install default-jre -y
RUN apt-get clean

# Copy necessary files
COPY . /work