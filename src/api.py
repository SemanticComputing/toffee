#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse
import logging

import eventlet
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO

from tasks import search_worker

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_timeout=600, message_queue='redis://')

log = logging.getLogger(__name__)

search_cache = dict()

eventlet.monkey_patch()


# TODO: Cache scrape results separately
# TODO: Cache everything to redis


@app.route('/')
def hello():
    return __doc__


@socketio.on('search')
def search(query):
    log.info('Search API got query: %s' % query)

    if query:
        search_worker.delay(query, request.sid, stopwords)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument("--host", default='127.0.0.1', help="Host (e.g. 127.0.0.1)")
    argparser.add_argument("--port", default=5000, help="Port (e.g. 5000)")
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = argparser.parse_args()

    stopwords = None
    with open('fin_stopwords.txt', 'r') as f:
        stopwords = f.read().split()

    with open('eng_stopwords.txt', 'r') as f:
        stopwords += f.read().split()

    stopwords += [str(num) for num in range(3000)]

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    socketio.run(app, host=args.host, port=args.port)
