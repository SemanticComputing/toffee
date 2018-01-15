#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search Celery tasks.
"""
import logging
import os
from hashlib import sha1
from operator import itemgetter

import eventlet
import redis
from celery import Celery
from collections import defaultdict
from flask import json
from flask_socketio import SocketIO

from search import RFSearch_GoogleAPI

eventlet.monkey_patch()

log = logging.getLogger(__name__)

apikey = os.environ['API_KEY']

prerender_host = os.environ.get('PRERENDER_HOST')
prerender_port = os.environ.get('PRERENDER_PORT')

redis_host = os.environ.get('REDIS_HOST', 'localhost')

search_cache = redis.StrictRedis(host=redis_host, port=6379, db=0)
scrape_cache = redis.StrictRedis(host=redis_host, port=6379, db=1)

celery_app = Celery('tasks', broker='redis://{host}'.format(host=redis_host))
socketio = SocketIO(message_queue='redis://{host}'.format(host=redis_host))


def search_cache_get(key, default=None):
    try:
        return json.loads(search_cache.get(key))
    except TypeError:
        return default


def search_cache_update(key, value, expire=60 * 60 * 24):
    return search_cache.setex(key, expire, json.dumps(value))


def fetch_results(searcher, words, sessionid):
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


def get_results(searcher, words, sessionid):
    results = fetch_results(searcher, words, sessionid)
    items = results['items']

    while words and not items:
        # Try to get items by removing the last words
        words.pop()

        results = fetch_results(searcher, words, sessionid)
        items = results['items']

    return results


@celery_app.task
def search_worker(query, sessionid, stopwords):
    socketio.emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])}, room=sessionid)
    search_words = query['data']['query'].split()
    log.debug('Got search words from API: {words}'.format(words=search_words))
    frontend_results = query['data'].get('results')
    log.debug('Got results from API: {res}'.format(res=frontend_results))

    searcher = RFSearch_GoogleAPI(apikey, stopwords=stopwords, scrape_cache=scrape_cache,
            prerender_host=prerender_host, prerender_port=prerender_port)

    results = get_results(searcher, search_words, sessionid)
    items = results['items']

    if not frontend_results:
        socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
        socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    results, topic_words = get_topics(searcher, results, sessionid)
    items = results['items']

    new_word_weights = defaultdict(int, zip(search_words, [1] * len(search_words)))
    if frontend_results:
        for item in items:
            url = item['url']
            thumb = next((res.get('thumb') for res in frontend_results if res.get('url') == url), None)

            if 'topic' not in item or not thumb:
                continue

            for topic, topic_weight in enumerate(item['topic']):
                for word, weight in topic_words[topic]:
                    weight = float(weight)
                    new_weight = topic_weight * weight * 500
                    log.info('Topic %s, word %s: %s' % (topic, word, new_weight))

                    new_word_weights[word] += weight * (1 if thumb else -1)

        new_search_words, _ = zip(*sorted(new_word_weights.items(), key=itemgetter(1), reverse=True))
        search_words = new_search_words[:(max(5, len(search_words)))]
        log.info('New search words based on topic modeling and thumbs: %s' % (new_search_words,))

    results = get_results(searcher, search_words, sessionid)
    items = results['items']

    socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    results, topic_words = get_topics(searcher, results, sessionid)
    items = results['items']

    socketio.emit('search_status_msg', {'data': 'Done'}, room=sessionid)
    results.update({'items': items, 'topic_words': topic_words})
