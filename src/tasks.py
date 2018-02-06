#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search Celery tasks.
"""
import eventlet; eventlet.monkey_patch()  # noqa

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

from arpa_linker.arpa import post

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

searcher = RFSearch_GoogleAPI(apikey, scrape_cache=scrape_cache, stopwords=stopwords,
                              prerender_host=prerender_host, prerender_port=prerender_port,
                              arpa_url=arpa_url)


def search_cache_get(key, default=None):
    if key is None:
        return default
    try:
        return json.loads(search_cache.get(key))
    except TypeError:
        return default


def search_cache_update(key, value, expire=60 * 60 * 24):
    if key is None:
        raise ValueError('Tried to update cache with None as key')
    return search_cache.setex(key, expire, json.dumps(value))


def get_query_hash(words):
    return sha1(' '.join(words).encode("utf-8")).hexdigest()


def fetch_results(words, sessionid):
    """
    Fetch results from cache or search class implementation.
    """
    socketio.emit('search_words', {'data': words}, room=sessionid)

    log.info('Fetch results words: {}'.format(words))

    query_hash = get_query_hash(words)
    cache_hit = search_cache_get(query_hash, {})
    items = cache_hit.get('items')

    if items:
        log.info('Cache hit for search id %s' % query_hash)
        # socketio.emit('search_words', {'data': cache_hit.get('words')}, room=sessionid)
        return cache_hit

    items = searcher.search(words, expand_words=False)

    log.debug('Got %s results through search' % len(items))

    results = {'query_hash': query_hash, 'items': items, 'words': words}
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
                text_content = cached_content.decode('utf-8')
                log.info('Found page content (%s chars) from scrape cache: %s' % (len(text_content), url))
        except TypeError:
            pass

    if not text_content:
        log.debug('Scraping document %s:  %s' % (item['name'], url))

        text_content = searcher.scrape(url)
        log.info('Scraped content length: {}'.format(len(text_content)))
        if scrape_cache and text_content:
            text_content = re.sub(r'\b\d+\b', '', text_content)
            text_content = re.sub(r'\s+', ' ', text_content)
            text_content = baseform_contents(text_content)
            log.info('Baseformed content length: {}'.format(len(text_content)))
            log.info('Adding page to scrape cache: %s' % (url))
            scrape_cache.setex(url, scrape_cache_expire, text_content)

    if text_content:
        item['contents'] = text_content

    return item


@celery_app.task
def get_topics(items, query_hash, sessionid):
    """
    Do topic modeling on search results, or retrieve from cache if found.
    """
    cache_hit = search_cache_get(query_hash, {})
    topic_words = cache_hit.get('topic_words')
    results = cache_hit

    log.info('Topic words: {}'.format(topic_words))

    if not topic_words:
        log.debug('Topic modeling for query hash %s' % query_hash)
        socketio.emit('search_status_msg', {'data': 'Topic modeling'}, room=sessionid)
        items, topic_words = searcher.topic_model(items)

        results.update({'items': items, 'topic_words': topic_words})
        search_cache_update(query_hash, results)

    log.info('Topic words: {}'.format(topic_words))

    return results, topic_words


@celery_app.task
def get_results(words, sessionid):
    log.debug('Get results with: {}, {}'.format(words, sessionid))

    results = fetch_results(words, sessionid)
    items = results['items']

    while words and not items:
        # Try to get items by removing the last words
        words = words[:-1]

        results = fetch_results(words, sessionid)
        items = results['items']

    socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    return items, results


def expand_words(words):
    words = searcher.filter_words(words)
    words = searcher.combine_expanded(searcher.word_expander(words))
    log.info('Expanded search words: {}'.format(words))

    return words


@celery_app.task
def refine_words(words, frontend_results, query_hash):
    """
    Refine the search query based on user feedback (thumbs up and down)
    """
    log.info('Refine words got initial words: {}'.format(words))
    if not len(frontend_results):
        return words

    cache_hit = search_cache_get(query_hash, {})
    topic_words = cache_hit.get('topic_words')
    items = cache_hit.get('items', [])

    new_word_weights = defaultdict(int, zip(words, [1] * len(words)))  # Initialized with old search words

    # Loop through each result and modify word weights based on its topics' words, if it has been thumbed
    for item in items:
        url = item['url']
        thumb = next((res.get('thumb') for res in frontend_results if res.get('url') == url), None)

        if 'topic' not in item or thumb is None:
            continue

        # Loop through topics and their words
        for topic, topic_weight in enumerate(item['topic']):
            for word, weight in topic_words[topic]:
                weight = topic_weight * float(weight) * 1000
                log.info('Topic %s, word %s: %s' % (topic, word, weight))

                new_word_weights[word] += weight * (1 if thumb else -5)

        # TODO: Expand new words to make them match old expanded words, use intersection to match

        new_search_words, _ = zip(*sorted(new_word_weights.items(), key=itemgetter(1), reverse=True))
        log.info('New search words based on topic modeling and thumbs: %s' %
                 sorted(new_word_weights.items(), key=itemgetter(1), reverse=True)[:10])
        words = new_search_words[:(max(5, len(words)))]

    return words


@celery_app.task
def emit_data_done(sessionid):
    socketio.emit('search_status_msg', {'data': 'Done'}, room=sessionid)


@celery_app.task
def combine_chunks(results):
    return [item for chunk in results for item in chunk]


def baseform_contents(text):
    data = {'text': text, 'locale': 'fi', 'depth': 0}
    query_result = post('http://las:9000/las/baseform', data, retries=3, wait=1)
    return query_result


@celery_app.task
def search_worker(query, sessionid):
    search_words = query['data'].get('words') or query['data']['query'].split()
    if len(search_words) == 0:
        return

    log.info('Got search words from API: {words}'.format(words=search_words))
    socketio.emit('search_status_msg', {'data': 'Search with {}'.format(query['data'])}, room=sessionid)

    frontend_results = query['data'].get('results') or {}
    log.debug('Got frontend results: {res}'.format(res=frontend_results))

    search_words = expand_words(search_words)
    query_hash = get_query_hash(search_words)

    refined_words = refine_words(search_words, frontend_results, query_hash)
    items, results = get_results(refined_words, sessionid)

    chain(scrape_page.chunks([(item, sessionid) for item in items], 10).group(),
            combine_chunks.s(),
            get_topics.s(query_hash, sessionid),
            emit_data_done.si(sessionid))()
