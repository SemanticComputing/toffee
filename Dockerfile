FROM python:jessie

RUN apt-get update \
    && apt-get install -y qt5-default libqt5webkit5-dev build-essential \
            python-lxml python-pip xvfb \
    && rm -rf /var/cache/apk/*

COPY src/requirements.txt /app/
WORKDIR /app/

RUN pip install -r requirements.txt

COPY src/ /app/

EXPOSE 5000

ENV FLASK_APP api.py

ARG API_KEY
ARG HOST="0.0.0.0"

ENV API_KEY ${API_KEY}
ENV HOST ${HOST}

CMD python api.py $API_KEY --host $HOST
