#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search Celery tasks.
"""
import logging
import os
from hashlib import sha1

import eventlet
import numpy as np
import redis
from celery import Celery
from flask import json, Flask
from flask_cors import CORS
from flask_socketio import SocketIO

from search import RFSearch_GoogleAPI

eventlet.monkey_patch()

log = logging.getLogger(__name__)

celery_app = Celery('tasks', broker='redis://')
socketio = SocketIO(message_queue='redis://')

search_cache = redis.StrictRedis(host='localhost', port=6379, db=0)


def search_cache_get(key, default=None):
    try:
        return json.loads(search_cache.get(key))
    except TypeError:
        return default


def search_cache_update(key, value):
    return search_cache.set(key, json.dumps(value))


def get_results(searcher, words, sessionid):
    query_hash = sha1(' '.join(words).encode("utf-8")).hexdigest()
    cache_hit = search_cache_get(query_hash, {})
    items = cache_hit.get('items')

    if items:
        log.info('Cache hit for search id %s' % query_hash)
        return cache_hit

    words = searcher.filter_words(words)
    expanded = searcher.combine_expanded(searcher.word_expander(words))
    socketio.emit('search_words', {'data': ' '.join(expanded)}, room=sessionid)
    items = searcher.search(expanded, expand_words=False)
    # items = pickle.load(open('google_search_results.pkl', 'rb'))
    log.debug('Got %s results through search' % len(items))

    results = {'result_id': query_hash, 'items': items}
    search_cache_update(query_hash, results)
    return results


def get_topics(searcher, results, sessionid):
    items = results.get('items')
    query_hash = results.get('result_id')

    cache_hit = search_cache_get(query_hash, {})
    topic_words = cache_hit.get('topic_words')

    if topic_words:
        return results, topic_words

    log.debug('Scraping for search id %s' % query_hash)
    socketio.emit('search_status_msg', {'data': 'Scraping'}, room=sessionid)
    items = searcher.scrape_contents(items)

    log.debug('Topic modeling for search id %s' % query_hash)
    socketio.emit('search_status_msg', {'data': 'Topic modeling'}, room=sessionid)
    items, topic_words = searcher.topic_model(items)

    results.update({'items': items, 'topic_words': topic_words})
    search_cache_update(query_hash, results)

    return results, topic_words


@celery_app.task
def search_worker(query, sessionid, stopwords):
    socketio.emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])}, room=sessionid)
    search_words = query['data']['query'].split()
    log.debug('Got search words from API: {words}'.format(words=search_words))
    frontend_results = query['data'].get('results')
    log.debug('Got results from API: {res}'.format(res=frontend_results))

    apikey = os.environ["APIKEY"]
    searcher = RFSearch_GoogleAPI(apikey, stopwords=stopwords)

    results = get_results(searcher, search_words, sessionid)
    items = results['items']

    if not frontend_results:
        socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
        socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    results, topic_words = get_topics(searcher, results, sessionid)
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

    # socketio.emit('search_words', {'data': search_words}, room=sessionid)

    results = get_results(searcher, search_words, sessionid)
    items = results['items']

    socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    results, topic_words = get_topics(searcher, results, sessionid)
    items = results['items']

    socketio.emit('search_status_msg', {'data': 'Done'}, room=sessionid)
    results.update({'items': items, 'topic_words': topic_words})
