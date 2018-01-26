#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""

import eventlet; eventlet.monkey_patch() # noqa

import argparse
import logging
import os

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO

from celery import Celery

app = Flask(__name__)
CORS(app)

redis_host = os.environ.get('REDIS_HOST', '')

celery_app = Celery('tasks', broker='redis://{host}'.format(host=redis_host),
        backend='redis://{host}'.format(host=redis_host))

socketio = SocketIO(app, ping_timeout=600, message_queue='redis://{host}'.format(host=redis_host))

log = logging.getLogger(__name__)

# TODO: Cache scrape results separately
# TODO: Cache everything to redis


@app.route('/')
def hello():
    return __doc__


@socketio.on('search')
def search(query):
    log.info('Search API got query: %s' % query)

    celery_app.send_task('tasks.search_worker', (query, request.sid))

    log.info('Called task')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument("--host", default='127.0.0.1', help="Host (e.g. 127.0.0.1)")
    argparser.add_argument("--port", default=5000, help="Port (e.g. 5000)")
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = argparser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log.info('API Redis host: {}'.format(redis_host))

    socketio.run(app, host=args.host, port=args.port)
