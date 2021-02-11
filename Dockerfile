FROM nvidia/cuda:10.2-devel

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip3 install scikit-build
RUN pip3 install cmake
RUN pip3 install torch torchvision opencv-python scipy flask
RUN pip3 install gunicorn

ENV FLASK_APP=api.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY . /app

CMD gunicorn --bind 0.0.0.0:5000 wsgi:app --access-logfile - --error-logfile -
