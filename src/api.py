#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse
import logging
import pickle
from hashlib import sha1

from flask import Flask, request, json
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from search import RFSearch_GoogleAPI

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)

log = logging.getLogger(__name__)

search_cache = dict()


@app.route('/')
def hello():
    return __doc__


@socketio.on('search')
def search(query):
    log.info('Search API got words: %s' % query)
    if query:
        emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])})
        # items = searcher.search(query['data'].split())
        items = pickle.load(open('google_search_results.pkl', 'rb'))
        emit('search_status_msg', {'data': 'Got {} results'.format(len(items))})
        log.info('Got %s results' % len(items))
        # pickle.dump(items, open('google_search_results.pkl', 'wb'))

        result_hash = sha1(json.dumps(items, sort_keys=True).encode("utf-8")).hexdigest()
        results = {'result_id': result_hash, 'items': items}
        emit('search_ready', {'data': json.dumps(results)})

        items = searcher.scrape_contents(items)
        items = searcher.topic_model(items)
        search_cache.update({result_hash: items})


@socketio.on('search_feedback')
def search_feedback(words, thumbs, result_id):
    log.info('Search feedback API got words: %s' % words)
    if words and result_id:
        previous_items = search_cache[result_id]

        query = words

        # TODO: Get words from previous result topics with thumbs up and add them to query

        # TODO: Remove words from previous result topics with thumbs down and add them to words

        search(query)


@socketio.on('my_broadcast_event')
def handle_message(message):
    print('received message: %s' % message)
    emit('my_response', message)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument("--host", default=None, help="Host (e.g. 0.0.0.0)")
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = argparser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    searcher = RFSearch_GoogleAPI(args.apikey)

    socketio.run(app, host=args.host)
