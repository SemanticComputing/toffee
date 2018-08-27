#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search using semantic knowledge and topic modeling.
"""
import argparse
import logging
import math
import re

import lda
import numpy as np
import requests
from arpa_linker.arpa import post
from bs4 import BeautifulSoup
from google import google
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer
from elasticsearch import Elasticsearch

from stop_words import STOP_WORDS

log = logging.getLogger(__name__)

# TODO: Query results from 2016+.


class RFSearch:
    """
    Relevance-feedback search abstract class.
    """

    def __init__(self,
            stopwords=None,
            search_cache=None,
            scrape_cache=None,
            topic_model=None,
            prerender_host='localhost',
            prerender_port='3000',
            arpa_url='http://demo.seco.tkk.fi/arpa/koko-related',
            baseform_url='http://demo.seco.tkk.fi/las/baseform'):
        """
        :param stopwords: list of stopwords
        :param scrape_cache: redis instance to use as a cache for web pages, or None for not using cache
        """

        if arpa_url:
            self.word_expander = SearchExpanderArpa(arpa_url=arpa_url).expand_words
        else:
            self.word_expander = lambda words: [(word,) for word in words]
        self.baseform_url = baseform_url
        self.stopwords = set(stopwords or [])  # Using set for better time complexity for "x in stopwords"
        self.search_cache = search_cache
        self.scrape_cache = scrape_cache
        self.topic_model = topic_model
        self.scrape_cache_expire = 60 * 60 * 24  # Expiry time in seconds
        self.prerender_host = prerender_host
        self.prerender_port = prerender_port

    @staticmethod
    def combine_expanded(expanded):
        return [' OR '.join(wordset) for wordset in expanded]

    def filter_words(self, words):
        filtered = [word for word in words if word not in self.stopwords]
        log.info('Stripped stop words: %s' % ((set(words) - set(filtered)) or '-'))
        return filtered

    def search(self, words):
        pass

    def baseform_contents(self, text):
        if not (text and self.baseform_url):
            return text
        data = {'text': text, 'locale': 'fi', 'depth': 0}
        query_result = post(self.baseform_url, data, retries=3, wait=1)
        return query_result

    def scrape(self, url, baseform=True):
        """
        Scrape a web page using prerender.

        :param url: URL
        :return: text contents, baseformed if baseform is truthy
        """

        if self.scrape_cache:
            page_content = self.scrape_cache.get_value(url)
            if page_content:
                log.info(
                    'Found page content (%s chars) from scrape cache: %s' % (len(page_content), url))
                return page_content.decode('utf-8')

        page = requests.get('http://{host}:{port}/{url}'.format(
            host=self.prerender_host,
            port=self.prerender_port,
            url=url))
        soup = BeautifulSoup(page.text, 'lxml')
        [s.extract() for s in soup(['iframe', 'script', 'style'])]

        page_content = soup.get_text()
        if page_content:
            page_content = re.sub(r'\b\d+\b', '', page_content)
            page_content = re.sub(r'\s+', ' ', page_content)
            log.info('Scraped {len} characters from URL: {url}'.format(len=len(page_content), url=url))
            if baseform:
                log.info('Baseforming content from URL: {url}'.format(len=len(page_content), url=url))
                page_content = self.baseform_contents(page_content)
                log.info('Baseformed content length: {len} characters from URL: {url}'.format(len=len(page_content), url=url))
            self.scrape_cache.set_value(url, page_content)
        else:
            log.warning('Unable to scrape any content for URL: %s' % url)

        return page_content

    def scrape_contents(self, documents):
        log.info('Scraping results')

        for doc in documents:
            text_content = None
            text_content = self.scrape(doc['url'])
            if text_content:
                doc['contents'] = text_content

        return documents


class TopicModeler:
    def __init__(self, stop_words=None, model=None, topic_words=None):
        self.stop_words = stop_words or []
        self.vectorizer = CountVectorizer(stop_words=list(self.stop_words))

        self.model = model
        self.topic_words = topic_words

    @property
    def doc_topic(self):
        return self.model.doc_topic_ if self.model else None

    def train(self, sample, n_topics=None):
        """
        Train the topic model with a list of text content

        >>> tm = TopicModeler()
        >>> tm.train(['something something dark side', 'jotain jotain pimeä puoli', 'something joke',
        ... 'tuota tätä muuta', 'something dark', 'tarve tuote myynti', 'Tampere Helsinki matkailu',
        ... 'uutinen turve tuotanto talous'])  # doctest: +ELLIPSIS
        [(('...', 0...), ('...', 0...), ('...'...))]

        >>> tm.topic_words # doctest: +ELLIPSIS
        [(('...', 0...), ('...', 0...), ('...'...))]

        >>> tm.doc_topic # doctest: +ELLIPSIS
        array([[0..., 0..., 0..., ...]])
        >>> len(tm.model.doc_topic_)
        8

        >>> tm.model.n_topics
        4
        """

        X = self.vectorizer.fit_transform(sample)
        vocab = self.vectorizer.get_feature_names()

        n_topics = n_topics or 1 + round(math.sqrt(len(sample)))

        self.model = lda.LDA(n_topics=n_topics, n_iter=2000, random_state=1)
        self.model.fit(X)

        n_top_words = 10
        eps = 0.001
        self.topic_words = []

        self.topic_words = self.get_top_topic_words(self.model.topic_word_, vocab, n_top_words, eps)

        return self.topic_words

    @staticmethod
    def get_top_topic_words(topic_words, vocab, n_top_words, eps):
        """
        >>> topic_words = np.array([[0.1, 0.2, 0.4, 0.25, 0.05], [0.2, 0.3, 0.2, 0.05, 0.35]])
        >>> vocab = ['sana', 'toinen', 'kolmas', 'neljäs', 'viides', 'kuudes']
        >>> TopicModeler.get_top_topic_words(topic_words, vocab, 3, 0.001)
        [(('kolmas', 0.4), ('neljäs', 0.25), ('toinen', 0.2)), (('viides', 0.35), ('toinen', 0.3), ('kolmas', 0.2))]
        """
        results = []
        for i, topic_dist in enumerate(topic_words):
            wordindex = np.argsort(topic_dist)[::-1]  # rev sort
            weights = topic_dist[wordindex]  # Topic word weights
            words = [np.array(vocab)[wordindex[j]] for j in range(min(n_top_words, len(wordindex))) if weights[j] > eps]
            word_weights = [weights[j] for j in range(min(n_top_words, len(wordindex))) if weights[j] > eps]
            log.debug('Topic {}: {}; {}'.format(i, ', '.join(words), ', '.join(map(str, word_weights))))
            results.append(tuple(zip(words, word_weights)))

        return results

    def get_topics(self, documents):
        """
        >>> import pprint
        >>> import numpy as np
        >>> tm = TopicModeler()
        >>> tm.train(['something something dark side', 'jotain jotain pimeä puoli', 'something joke',
        ... 'tuota tätä muuta puoli', 'something dark', 'tarve tuote myynti jotain muuta',
        ... 'Tampere Helsinki matkailu uutinen', 'uutinen turve tuotanto matkailu'], n_topics=3) # doctest: +ELLIPSIS
        [((...))]
        >>> corpus = [{'url': 'http...', 'contents': 'nah bro'},
        ... {'url': 'http1', 'contents': 'jotain uutinen Helsinki'},
        ... {'url': 'http2', 'contents': 'something dark'},
        ... {'url': 'http0', 'contents': 'Helsinki tuotanto'},
        ... {'url': 'http-1', 'contents': 'kameli järvi tuote'},
        ... {'url': 'http-2', 'contents': 'hevonen talikko navetta'},
        ... {'url': 'http3', 'contents': 'muu talous tarvike työkalu maailma'}]
        >>> topics = tm.get_topics(corpus)
        >>> pprint.pprint(topics) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        ([{'contents': 'nah bro', 'topic': [..., ..., ...], 'url': 'http1'}, ...], [((...))])
        >>> pprint.pprint(topics[0][2]) # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        {'contents': 'something dark', 'topic': [...], 'url': 'http2'}
        >>> topic = topics[1][np.argmax(topics[0][2]['topic'])]
        >>> 'something' in [t[0] for t in topic]
        True
        >>> 'dark' in [t[0] for t in topic]
        True
        """

        X = self.vectorizer.transform([r.get('contents', '') for r in documents])
        for (doc, topic) in zip(documents, self.model.transform(X)):
            doc['topic'] = topic.tolist()
            log.debug(doc.get('name'))
            log.debug(self.topic_words[np.argmax(topic)])

        return documents, self.topic_words


class SearchExpanderArpa:
    """
    Search expander class using ARPA.
    """

    def __init__(self, arpa_url='http://demo.seco.tkk.fi/arpa/koko-related'):
        self.arpa_url = arpa_url

    def expand_words(self, words):
        """
        ARPA query expansion using related items from KOKO ontology.

        :param words: list of words
        :return: list of tuples containing words with their related words
        """
        log.debug('Querying ARPA')

        expanded = []
        for word in words:
            if ' OR ' in word:
                log.info('Skipping expansion for already expanded term {}'.format(word))
                expanded.append((word,))
                continue

            data = {'text': word}
            log.info('Expanding {}'.format(word))
            query_result = post(self.arpa_url, data, retries=3, wait=1)

            related = [match['properties'].get('relatedlabel', '') for match in query_result['results']]
            related = [item for sublist in related for item in set(sublist)]
            related = [x if ' ' in x else x.strip('"') for x in related]

            related = [word] + [x for x in related if x != word]
            expanded.append(related)

        log.info('Expanded from %s words to %s words' % (len(words), len([x for y in expanded for x in y])))

        return expanded


class RFSearchGoogleAPI(RFSearch):
    """
    Relevance-feedback search using Google Custom Search.
    """

    def __init__(self,
                 apikey='',
                 **kwargs):

        super().__init__(**kwargs)
        self.search_service = build("customsearch", "v1", developerKey=apikey)
        if self.topic_model is None:
            self.topic_model = TopicModeler(stop_words=STOP_WORDS)

    def train_topic_model(self, documents):
        data_corpus = [r.get('contents', '') for r in documents]
        self.topic_model.train(data_corpus)

    def format_query(self, words):
        return ' '.join(words)

    def baseform_document(self, item):
        item['contents'] = self.baseform_contents(item.get('contents'))
        return item

    def search(self, words, expand_words=True):
        """
        Create a search query based on a list of words and query for results.

        :param words: list of words
        :param expand_words:
        :return:
        """
        if expand_words:
            words = self.combine_expanded(self.word_expander(words))

        query = self.format_query(words)
        while len(query) > 2500:
            words.pop()
            query = ' '.join(words)

        log.info('Query: %s' % query)

        if self.search_cache:
            cache_hit = self.search_cache.get_json(query)
            if cache_hit:
                log.info('Search cache hit')
                return cache_hit
            log.info('Query not found from search cache')

        res = self.search_service.cse().list(
            q=query,
            cx='012121639191539030590:cshq4wzc7ms',
            num=10,
            start=1,
        ).execute()

        items = res.get('items') or []
        next_page = True
        i = 1

        total_results = int(res.get('searchInformation').get('totalResults'))
        log.debug('Got %s total results from initial search query' % total_results)

        while total_results > 10 and next_page and i < 40:
            i += 10
            res = self.search_service.cse().list(
                q=query,
                cx='012121639191539030590:cshq4wzc7ms',
                num=10,
                start=i,
            ).execute()

            new_items = res.get('items')

            if new_items:
                items += new_items
            else:
                next_page = False

            if i + 10 > total_results:
                next_page = False

        sanitized = []
        for item in items:
            document = {'name': item['title'], 'url': item['link'], 'description': item['snippet']}
            sanitized.append(document)

        results = len(sanitized)
        sanitized = (sanitized, None)

        if results and self.search_cache:
            self.search_cache.set_json(query, sanitized)

        log.info('Returning %s results from search.' % results)

        return sanitized


class RFSearchGoogleUI(RFSearch):
    """
    Relevance-feedback search using Google Custom Search. Outdated.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.num_results = 10

    @staticmethod
    def combine_expanded(words):
        return [' OR '.join(wordset) for wordset in words]

    def search(self, words, expand_words=True):
        """
        Create a search query based on a list of words and query for results.

        :param expand_words:
        :param words:
        :return:
        """
        if expand_words:
            words = self.combine_expanded(self.word_expander(words))

        query = ' '.join(words)

        log.info('Query: %s' % query)

        res = google.search(query, self.num_results)

        sanitized = []
        for item in res:
            sanitized.append({'name': item.name, 'url': item.link, 'description': item.description})

        log.info('Got %s results from search.' % len(sanitized))

        return sanitized, None


class RFSearchElastic(RFSearch):
    """
    Relevance-feedback search using Elasticsearch.
    """

    def __init__(self,
                 elastic_nodes=[{'host': 'elastic', 'port': 9200}],
                 elastic_index='ylenews',
                 **kwargs):

        super().__init__(**kwargs)
        self.es = Elasticsearch(elastic_nodes)
        self.es_index = elastic_index

    def baseform_document(self, item):
        item['contents'] = self.baseform_contents(item.get('contents'))
        return item

    @staticmethod
    def format_query(words):
        return '({})'.format(') ('.join(words))

    def search(self, words, expand_words=False, baseform=True):
        """
        Create a search query based on a list of words and query for results.

        :param words: list of words
        :param expand_words:
        :return:
        """
        query = self.format_query(words)

        log.info('Query: {query}'.format(query=query))

        if self.search_cache:
            cache_hit = self.search_cache.get_json(query)
            if cache_hit:
                log.info('Search cache hit')
                return cache_hit
            log.info('Query not found from search cache')

        # Make search
        res = self.es.search(index=self.es_index, body={"size": 100, "query": {"query_string": {"query": query}}})

        hits = res.get('hits', {})
        results = hits.get('hits', [])

        total_results = hits.get('total')
        log.debug('Got {total_results} total results from initial search query'.format(total_results=total_results))

        sanitized = []
        for item in results:
            source = item.get('_source')
            url = source.get('url', {})
            text_content = '\n\n'.join((cont.get('text', cont.get('alt', '')) for cont in source.get('content')))
            document = {
                'name': source.get('headline', {}).get('full'),
                'url': url.get('short', url.get('full')),
                'contents': text_content,
                'description': source.get('lead', '')
            }
            if baseform:
                document = self.baseform_document(document)
            sanitized.append(document)

        sanitized = self.topic_model.get_topics(sanitized)
        if sanitized and self.search_cache:
            self.search_cache.set_json(query, sanitized)

        return sanitized


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("--engine", help="Search engine", default='GoogleAPI',
                           choices=["GoogleAPI", "GoogleUI", "Elastic"])
    argparser.add_argument("--apikey", help="Google API key")
    argparser.add_argument('words', metavar='Keywords', type=str, nargs='+', help='search keywords')
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = argparser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ENGINES = {'GoogleAPI': RFSearchGoogleAPI,
               'GoogleUI': RFSearchGoogleUI,
               'Elastic': RFSearchElastic
               }

    apikey = args.apikey
    search_words = args.words
    engine = args.engine

    np.set_printoptions(precision=3, suppress=True)

    # with open('stopwords.txt', 'r') as f:
    #     stopwords = f.read().split()
    #
    params = {}
    # params.update({'stopwords': stopwords})
    if apikey:
        params.update({'apikey': apikey})
    searcher = ENGINES[engine](**params)

    docs = searcher.search(search_words)
    # docs = pickle.load(open('google_search_results.pkl', 'rb'))
    # pickle.dump(res, open('google_search_results.pkl', 'wb'))

    if 'contents' not in docs[0]:
        docs = searcher.scrape_contents(docs)

    docs, _ = searcher.topic_model(docs)

    for doc in docs:
        print(doc['name'].upper())
        print(doc['description'])
        print(doc.get('topic'))
        print()
