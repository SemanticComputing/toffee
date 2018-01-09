#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse
import logging
import pickle
from hashlib import sha1

import eventlet
import numpy as np
from flask import Flask, request, json
from flask_cors import CORS
from flask_socketio import SocketIO, send, emit

from search import RFSearch_GoogleAPI

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, ping_timeout=600, message_queue='redis://')

log = logging.getLogger(__name__)

search_cache = dict()

eventlet.monkey_patch()

# TODO: Cache scrape results separately


def get_results(words):
    query_hash = sha1(' '.join(words).encode("utf-8")).hexdigest()
    cache_hit = search_cache.get(query_hash, {})
    items = cache_hit.get('items')

    if items:
        log.info('Cache hit for search id %s' % query_hash)
        return cache_hit

    items = searcher.search(words)
    # items = pickle.load(open('google_search_results.pkl', 'rb'))
    log.debug('Got %s results through search' % len(items))

    results = {'result_id': query_hash, 'items': items}
    search_cache.update({query_hash: results})
    return results


def get_topics(results):
    items = results.get('items')
    query_hash = results.get('result_id')

    cache_hit = search_cache.get(query_hash, {})
    topic_words = cache_hit.get('topic_words')

    if topic_words:
        return results, topic_words

    log.debug('Scraping for search id %s' % query_hash)
    emit('search_status_msg', {'data': 'Scraping'})
    items = searcher.scrape_contents(items)

    log.debug('Topic modeling for search id %s' % query_hash)
    emit('search_status_msg', {'data': 'Topic modeling'})
    items, topic_words = searcher.topic_model(items)

    results.update({'items': items, 'topic_words': topic_words})
    search_cache.update({query_hash: results})

    return results, topic_words


@app.route('/')
def hello():
    return __doc__


@socketio.on('search')
def search(query):
    log.info('Search API got query: %s' % query)

    if query:
        emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])})
        search_words = query['data']['query'].split()
        frontend_results = query['data'].get('results')

        results = get_results(search_words)
        items = results['items']

        emit('search_words', {'data': search_words})

        if not frontend_results:
            emit('search_status_msg', {'data': 'Got {} results'.format(len(items))})
            emit('search_ready', {'data': json.dumps(results)})

        results, topic_words = get_topics(results)
        items = results['items']

        if frontend_results:
            for item in items:
                url = item['url']
                thumb = next((res.get('thumb') for res in frontend_results if res.get('url') == url), None)
                if 'topic' in item:
                    thumb_words = topic_words[np.argmax(item['topic'])]
                    if thumb is True:
                        # TODO: Make magic happen
                        new_words = thumb_words[:3]
                        search_words += [word for word in new_words if word not in search_words]
                    elif thumb is False:
                        [search_words.remove(word) for word in thumb_words if word in search_words]

            search_words = search_words[:20]

        emit('search_words', {'data': search_words})

        results = get_results(search_words)
        items = results['items']

        emit('search_status_msg', {'data': 'Got {} results'.format(len(items))})
        emit('search_ready', {'data': json.dumps(results)})

        results, topic_words = get_topics(results)
        items = results['items']

        emit('search_status_msg', {'data': 'Done'})
        results.update({'items': items, 'topic_words': topic_words})

        log.info('Dumping pickle')
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
