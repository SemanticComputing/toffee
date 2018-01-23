#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search Celery tasks.
"""
import eventlet; eventlet.monkey_patch() # noqa

import logging
import os
import re
from hashlib import sha1
from operator import itemgetter

import redis
from celery import Celery, chain
from collections import defaultdict
from flask import json
from flask_socketio import SocketIO

from search import RFSearch_GoogleAPI


log = logging.getLogger(__name__)

apikey = os.environ['API_KEY']

prerender_host = os.environ.get('PRERENDER_HOST')
prerender_port = os.environ.get('PRERENDER_PORT')

arpa_url = os.environ.get('ARPA_URL')

redis_host = os.environ.get('REDIS_HOST', 'localhost')

search_cache = redis.StrictRedis(host=redis_host, port=6379, db=0)
scrape_cache = redis.StrictRedis(host=redis_host, port=6379, db=1)

scrape_cache_expire = 60 * 60 * 24  # Expiry time in seconds

celery_app = Celery('tasks', broker='redis://{host}'.format(host=redis_host),
        backend='redis://{host}'.format(host=redis_host))
socketio = SocketIO(message_queue='redis://{host}'.format(host=redis_host))

stopwords = None
with open('fin_stopwords.txt', 'r') as f:
    stopwords = f.read().split()

with open('eng_stopwords.txt', 'r') as f:
    stopwords += f.read().split()

stopwords += [str(num) for num in range(3000)]

searcher = RFSearch_GoogleAPI(apikey, scrape_cache=scrape_cache, stopwords=stopwords,
                              prerender_host=prerender_host, prerender_port=prerender_port,
                              arpa_url=arpa_url)


def search_cache_get(key, default=None):
    try:
        return json.loads(search_cache.get(key))
    except TypeError:
        return default


def search_cache_update(key, value, expire=60 * 60 * 24):
    return search_cache.setex(key, expire, json.dumps(value))


def fetch_results(words, sessionid):
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


@celery_app.task
def scrape_page(item, sessionid):
    log.info('Scrape: {}'.format(item))
    url = item['url']
    log.debug('Scraping URL %s' % url)
    socketio.emit('search_status_msg', {'data': 'Scraping'}, room=sessionid)

    text_content = None
    if scrape_cache:
        try:
            cached_content = scrape_cache.get(url)
            if cached_content:
                text_content = str(cached_content)
                log.info('Found page content (%s chars) from scrape cache: %s' % (len(text_content), url))
        except TypeError:
            pass

    if not text_content:
        log.debug('Scraping document %s:  %s' % (item['name'], url))

        text_content = searcher.scrape(url)
        if scrape_cache and text_content:
            text_content = re.sub(r'\s+', ' ', text_content)
            log.info('Adding page to scrape cache: %s' % (url))
            scrape_cache.setex(url, scrape_cache_expire, text_content)

    if text_content:
        item['contents'] = text_content

    return item


@celery_app.task
def get_topics(items, result_id, sessionid):
    cache_hit = search_cache_get(result_id, {})
    topic_words = cache_hit.get('topic_words')
    results = cache_hit

    if not topic_words:
        log.debug('Topic modeling for search id %s' % result_id)
        socketio.emit('search_status_msg', {'data': 'Topic modeling'}, room=sessionid)
        items, topic_words = searcher.topic_model(items)

        results.update({'items': items, 'topic_words': topic_words})
        search_cache_update(result_id, results)

    return results, topic_words


@celery_app.task
def get_results(words, sessionid):
    log.debug('Get results with: {}, {}'.format(words, sessionid))
    results = fetch_results(words, sessionid)
    items = results['items']

    while words and not items:
        # Try to get items by removing the last words
        words.pop()

        results = fetch_results(words, sessionid)
        items = results['items']

    socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    return items, results


@celery_app.task
def refine_words(search_words, frontend_results, result_id):
    if not len(frontend_results):
        return search_words

    cache_hit = search_cache_get(result_id, {})
    topic_words = cache_hit.get('topic_words')
    items = cache_hit.get('items')

    new_word_weights = defaultdict(int, zip(search_words, [1] * len(search_words)))
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

    return search_words


@celery_app.task
def emit_data_done(sessionid):
    socketio.emit('search_status_msg', {'data': 'Done'}, room=sessionid)


@celery_app.task
def combine_chunks(results):
    return [item for chunk in results for item in chunk]


@celery_app.task
def search_worker(query, sessionid):
    search_words = query['data']['query'].split()
    log.info('Got search words from API: {words}'.format(words=search_words))
    socketio.emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])}, room=sessionid)

    frontend_results = query['data'].get('results') or {}
    log.debug('Got frontend results: {res}'.format(res=frontend_results))

    result_id = query['data'].get('result_id')

    refined_words = refine_words(search_words, frontend_results, result_id)
    items, results = get_results(refined_words, sessionid)

    chain(scrape_page.chunks([(item, sessionid) for item in items], 10).group(),
            combine_chunks.s(),
            get_topics.s(result_id, sessionid),
            emit_data_done.si(sessionid))()
