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
        search_words = query['data']['query'].split()
        query_hash = sha1(' '.join(search_words).encode("utf-8")).hexdigest()

        items = search_cache.get(query_hash, {}).get('items')

        if not items:
            items = searcher.search(search_words)
            # items = pickle.load(open('google_search_results.pkl', 'rb'))
            log.info('Got %s results through search' % len(items))
        else:
            log.info('Got %s results from cache' % len(items))

        emit('search_status_msg', {'data': 'Got {} results'.format(len(items))})

        # TODO: Thumbz

        results = {'result_id': query_hash, 'items': items}

        emit('search_ready', {'data': json.dumps(results)})

        emit('search_status_msg', {'data': 'Processing results'})
        topics = search_cache.get(query_hash, {}).get('has_topics')

        if topics:
            results.update({'topics': topics})
            log.info('Cache hit for search id %s' % query_hash)
        else:
            log.info('Scraping for search id %s' % query_hash)
            emit('search_status_msg', {'data': 'Scraping'})
            items = searcher.scrape_contents(items)

            log.info('Topic modeling for search id %s' % query_hash)
            emit('search_status_msg', {'data': 'Topic modeling'})
            items = searcher.topic_model(items)

        emit('search_status_msg', {'data': 'Done'})
        results = {'result_id': query_hash, 'items': items, 'has_topics': True}

        log.info('Updating cache for search id %s' % query_hash)
        search_cache.update({query_hash: results})
        pickle.dump(search_cache, open('search_cache.pkl', 'wb'))


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

    if not search_cache:
        try:
            search_cache = pickle.load(open('search_cache.pkl', 'rb'))
        except FileNotFoundError:
            log.info('Search cache log file not found.')

    socketio.run(app, host=args.host)
