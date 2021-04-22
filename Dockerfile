FROM tensorflow/tensorflow:2.4.1

COPY ./requirements.txt /requirements.txt
RUN apt-get update
RUN apt-get install -y python3.8 screen
RUN rm /usr/bin/python3
RUN ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python -m pip install --upgrade pip
RUN python -m pip install --default-timeout=100 -r /requirements.txt

WORKDIR /work
CMD make boot-app && /bin/bash