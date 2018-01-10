FROM python:3

RUN apt-get update \
    && apt-get install -y python-lxml python-pip \
    && rm -rf /var/cache/apk/*

COPY src/requirements.txt /app/
WORKDIR /app/

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY src/ /app/

EXPOSE 5000

ENV FLASK_APP api.py

ARG API_KEY
ARG HOST="0.0.0.0"
ARG PORT="5000"

ENV API_KEY ${API_KEY}
ENV HOST ${HOST}
ENV PORT ${PORT}

CMD python api.py $API_KEY --host $HOST --port $PORT
