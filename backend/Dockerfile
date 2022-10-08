FROM python:3.9

RUN apt-get update \
  && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential

WORKDIR /app

RUN curl "https://www.dropbox.com/s/yx6n606i7cfcvoz/WilhemNet_86.h5?dl=1" -L -o WilhemNet_86.h5

COPY backend/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY backend/src .
COPY protos/backend.proto .

ADD https://github.com/afrincon/neumonia_detector/releases/download/model/WilhemNet_86.h5 /app

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./backend.proto