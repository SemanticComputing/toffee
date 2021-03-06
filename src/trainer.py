#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import joblib
import logging

import math
from elasticsearch import Elasticsearch

from search import TopicModeler, RFSearch
from stop_words import STOP_WORDS

if __name__ == '__main__':
    logging.basicConfig(level=getattr(logging, 'INFO'),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    log = logging.getLogger(__name__)

    ELASTIC_HOST = os.environ.get('ELASTIC_HOST', 'elastic')
    ELASTIC_PORT = os.environ.get('ELASTIC_PORT', '9200')

    es = Elasticsearch([{'host': ELASTIC_HOST, 'port': ELASTIC_PORT}])
    es_index = 'ylenews'

    query = {
        'size': 10000,
        '_source': ['content.text', 'content.alt'],
        'query': {
            'function_score': {
                'query': {'match_all': {}},
                'random_score': {}
            }
        }
    }

    bf_url = os.environ.get('BASEFORM_URL')
    bf_args = {'baseform_url': bf_url} if bf_url else {}

    bf = RFSearch(**bf_args)

    log.info('Getting sample...')
    res = es.search(index=es_index, body=query)['hits']['hits']

    log.info('Baseforming sample content...')
    text_content = [bf.baseform_contents(' '.join([cont.get('text', cont.get('alt', ''))
                                                   for cont in article['_source']['content']])) for article in res]

    tm = TopicModeler(stop_words=STOP_WORDS)

    n_topics = round(math.sqrt(100 * len(text_content)))
    log.info('Training topic model with {n} topics...'.format(n=n_topics))
    tm.train(text_content, n_topics=n_topics)

    log.info('Dumping topic model...')
    joblib.dump(tm, 'output/topic_model.pkl')

    log.info('Done')
