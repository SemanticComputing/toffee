#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search Celery tasks.
"""
import eventlet;

eventlet.monkey_patch()  # noqa

import logging
import os

import joblib
import redis
from operator import itemgetter
from celery import Celery, chain
from collections import defaultdict
from flask import json
from flask_socketio import SocketIO

from search import RFSearchGoogleAPI, RFSearchElastic
from stop_words import STOP_WORDS

log = logging.getLogger(__name__)

APIKEY = os.environ['API_KEY']
PRERENDER_HOST = os.environ.get('PRERENDER_HOST')
PRERENDER_PORT = os.environ.get('PRERENDER_PORT')
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'elastic')
ELASTIC_PORT = os.environ.get('ELASTIC_PORT', '9200')
ARPA_URL = os.environ.get('ARPA_URL')
BASEFORM_URL = os.environ.get('BASEFORM_URL')


class Cache:
    def __init__(self, redis_host, db):
        self.cache = redis.StrictRedis(host=redis_host, port=6379, db=db)

    def get_value(self, key, default=None):
        if key is None:
            return default
        return self.cache.get(key)

    def set_value(self, key, value, expire=60 * 60 * 24):
        if key is None:
            raise ValueError('Tried to update cache with None as key')
        return self.cache.setex(key, expire, value)

    def set_json(self, key, value):
        return self.set_value(key, json.dumps(value))

    def get_json(self, key, default=None):
        try:
            return json.loads(self.get_value(key))
        except TypeError:
            return default


search_cache_google = Cache(REDIS_HOST, db=0)
scrape_cache_google = Cache(REDIS_HOST, db=1)
search_cache_elastic = Cache(REDIS_HOST, db=2)

try:
    topic_model = joblib.load('topics/topic_model.pkl')
except FileNotFoundError:
    topic_model = None

searcher_google = RFSearchGoogleAPI(APIKEY,
                                    stopwords=STOP_WORDS,
                                    prerender_host=PRERENDER_HOST, prerender_port=PRERENDER_PORT,
                                    search_cache=search_cache_google,
                                    scrape_cache=scrape_cache_google,
                                    arpa_url=ARPA_URL,
                                    baseform_url=BASEFORM_URL)

searcher_elastic = RFSearchElastic(elastic_nodes=[{'host': ELASTIC_HOST, 'port': ELASTIC_PORT}],
                                   stopwords=STOP_WORDS,
                                   search_cache=search_cache_elastic,
                                   arpa_url=ARPA_URL,
                                   topic_model=topic_model,
                                   baseform_url=BASEFORM_URL)

celery_app = Celery('tasks', broker='redis://{host}'.format(host=REDIS_HOST),
                    backend='redis://{host}'.format(host=REDIS_HOST))
socketio = SocketIO(message_queue='redis://{host}'.format(host=REDIS_HOST))


@celery_app.task
def scrape_page(url, sessionid):
    """
    Scrape a URL. Only needed with Google.
    """

    log.info('Scrape: {}'.format(url))
    socketio.emit('search_status_msg', {'data': 'Scraping'}, room=sessionid)

    text_content = searcher_google.scrape(url)
    log.info('Scraped content length: {}'.format(len(text_content)))

    return {'url': url, 'contents': text_content}


def fetch_results(words, sessionid, searcher):
    """
    Fetch results from cache or search class implementation, based on the search words.
    """

    socketio.emit('search_words', {'data': words}, room=sessionid)

    log.info('Fetch results words: {}'.format(words))

    items, topic_words = searcher.search(words, expand_words=False)

    log.debug('Got %s results through search' % len(items))

    results = {'items': items, 'words': words, 'topic_words': topic_words}

    return results


def get_results(words, sessionid, searcher):
    """
    Get results for the given search words and emit them to client
    """
    log.info('Get results with: {}, {}'.format(words, sessionid))

    results = fetch_results(words, sessionid, searcher)
    items = results['items']

    while words and not items:
        # Try to get items by removing the last words
        words = words[:-1]

        results = fetch_results(words, sessionid, searcher)
        items = results['items']

    socketio.emit('search_status_msg', {'data': 'Got {} results'.format(len(items))}, room=sessionid)
    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    return items, results


def expand_words(words, banned_words, searcher):
    """
    >>> from unittest.mock import MagicMock
    >>> searcher.filter_words = MagicMock(side_effect=lambda x: x)
    >>> searcher.word_expander = MagicMock(side_effect=lambda words: [(word,) for word in words])

    >>> words = ['innovaatio', 'teknologia']
    >>> expand_words(words, [])
    ['innovaatio', 'teknologia']

    >>> words = ['innovaatio OR something', 'teknologia']
    >>> expand_words(words, [])
    ['innovaatio OR something', 'teknologia']

    >>> words = ['innovaatio OR something', 'teknologia']
    >>> expand_words(words, ['innovaatio'])
    ['something', 'teknologia']
    >>> expand_words(words, ['teknologia'])
    ['innovaatio OR something']

    >>> searcher.word_expander = MagicMock(side_effect=lambda words: [(word, 'other') for word in words])
    >>> words = ['innovaatio', 'teknologia']
    >>> expand_words(words, [])
    ['innovaatio OR other', 'teknologia OR other']
    >>> expand_words(words, ['other'])
    ['innovaatio', 'teknologia']
    >>> searcher.word_expander = MagicMock(side_effect=lambda words: [tuple([word] + ['o%s' % x for x in range(10)]) for word in words])
    >>> expand_words(words, [])
    ['innovaatio OR o0 OR o1 OR o2 OR o3 OR o4', 'teknologia OR o0 OR o1 OR o2 OR o3 OR o4']
    >>> expand_words(words, ['o2', 'o4'])
    ['innovaatio OR o0 OR o1 OR o3 OR o5 OR o6', 'teknologia OR o0 OR o1 OR o3 OR o5 OR o6']
    """

    log.info(
        'Expand words: {words}; remove banned words: {banned_words}'.format(words=words, banned_words=banned_words))
    words = searcher.filter_words(words)
    words = searcher.word_expander(words)
    filtered_words = []
    for chunk in words:
        chunk = [word for w in chunk for word in w.split(' OR ') if word not in banned_words][:6]
        if chunk:
            filtered_words.append(chunk)
    words = filtered_words

    words = searcher.combine_expanded(words)
    log.info('Expanded search words: {}'.format(words))

    return words


def refine_words(words, frontend_query, searcher):
    """
    Refine the search query based on user feedback (thumbs up and down)
    """

    log.info('Refine words got initial words: {}'.format(words))

    frontend_results = frontend_query.get('results') if frontend_query else None
    if not frontend_results:
        log.info('No thumbs received')
        return words

    documents, topic_words = searcher.search(words)
    new_word_weights = defaultdict(int, zip(words, [1] * len(words)))  # Initialized with old search words

    # Loop through each result and modify word weights based on its topics' words, if it has been thumbed
    for result in [res for res in frontend_results if res.get('thumb') is not None]:
        url = result['url']
        thumb = result['thumb']
        topics = next((document.get('topic') for document in documents if document['url'] == url), None)
        log.info('Weighting {url} {thumb} {topics}'.format(url=url, thumb=thumb, topics=topics))

        if not topics:
            log.warning('No topics found for {}'.format(url))
            continue

        # Loop through topics and their words
        for topic, topic_weight in enumerate(topics):
            for word, weight_in_topic in topic_words[topic]:
                weight = topic_weight * float(weight_in_topic) * 50
                log.debug('Topic %s, word %s: %.10f -> %.10f' % (topic, word, weight_in_topic, weight))

                # Match word to existing expanded words in a non-robust way:
                for existing in new_word_weights.keys():
                    if word in existing.split(' OR '):
                        word = existing

                new_word_weights[word] += weight * (1 if thumb else -1)

    new_search_words, weights = zip(*sorted(new_word_weights.items(), key=itemgetter(1), reverse=True))
    num_words = max(5, len(words))
    log.info('Top 50 refined search words (%s used): %s' % (num_words, list(zip(new_search_words, weights))[:50]))
    words = new_search_words[:num_words]

    return words


@celery_app.task
def emit_data_done(sessionid):
    socketio.emit('search_status_msg', {'data': 'Done'}, room=sessionid)
    socketio.emit('search_processing_finished', {'data': None}, room=sessionid)


@celery_app.task
def combine_chunks(results, items):
    if not results:
        return items
    if type(results[0]) != dict:
        results = [item for chunk in results for item in chunk]
    results = {item['url']: item['contents'] for item in results}
    for item in items:
        item['contents'] = results.get(item['url'])
    return items


@celery_app.task
def topic_model_documents(items, results, sessionid, words):
    """
    Do real-time topic modeling on search results, or retrieve from cache if found.
    """
    socketio.emit('search_status_msg', {'data': 'Topic modeling'}, room=sessionid)

    searcher_impl = searcher_google
    searcher_impl.train_topic_model(items)

    items, topic_words = searcher_impl.topic_model.get_topics(items)
    log.info('Topic words: {}'.format(topic_words))
    results.update({'items': items, 'topic_words': topic_words})

    socketio.emit('search_ready', {'data': json.dumps(results)}, room=sessionid)

    # Update search cache
    log.info('Updating cache with topics for words: {}'.format(words))
    searcher_impl.search_cache.set_json(searcher_impl.format_query(words), (items, topic_words))

    return results, topic_words


def parse_query(query, sessionid):
    """
    Parse search words from query object received from the UI.
    """
    log.info('Got frontend query: {query}'.format(query=query))
    search_words = query['data'].get('words') or query['data']['query'].split()
    if '()' in search_words:
        search_words.remove('')
    if not search_words:
        return

    banned_words = query['data'].get('banned_words')

    log.info('Got search words from API: {words}'.format(words=search_words))

    socketio.emit('search_status_msg', {'data': 'Searching'}, room=sessionid)
    return search_words, banned_words


@celery_app.task
def search_worker_google(query, sessionid):
    """
    Initiate search iteration using google search.
    """
    search_words, banned_words = parse_query(query, sessionid)

    log.info('search_worker_google search words: {}'.format(search_words))
    refined_words = refine_words(search_words, query['data'], searcher_google)
    refined_words = expand_words(refined_words, banned_words, searcher_google)

    log.info('search_worker_google refined search words: {}'.format(refined_words))
    items, results = get_results(refined_words, sessionid, searcher_google)

    chain(scrape_page.chunks([(item['url'], sessionid) for item in items], 20).group(),
          combine_chunks.s(items),
          topic_model_documents.s(results, sessionid, refined_words),
          emit_data_done.si(sessionid))()


@celery_app.task
def search_worker_elastic(query, sessionid):
    """
    Initiate search iteration using elasticsearch.
    """
    search_words, banned_words = parse_query(query, sessionid)

    refined_words = refine_words(search_words, query['data'], searcher_elastic)
    refined_words = expand_words(refined_words, banned_words, searcher_elastic)

    items, results = get_results(refined_words, sessionid, searcher_elastic)

    emit_data_done(sessionid)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
