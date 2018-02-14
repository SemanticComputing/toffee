#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search using semantic knowledge and topic modeling.
"""
import argparse
import logging

import math
import lda
import numpy as np
import requests
from arpa_linker.arpa import post
from bs4 import BeautifulSoup
from google import google
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import CountVectorizer

log = logging.getLogger(__name__)


# TODO: Query results from 2016+.


class RFSearch:
    """
    Relevance-feedback search abstract class.
    """

    def __init__(self, stopwords=None, scrape_cache=None, prerender_host='localhost',
            prerender_port='3000'):
        """
        :param stopwords: list of stopwords
        :param scrape_cache: redis instance to use as a cache for web pages, or None for not using cache
        """

        self.stopwords = set(stopwords or [])  # Using set for better time complexity for "x in stopwords"
        self.scrape_cache = scrape_cache
        self.scrape_cache_expire = 60 * 60 * 24  # Expiry time in seconds
        self.prerender_host = prerender_host
        self.prerender_port = prerender_port

    def filter_words(self, words):
        filtered = [word for word in words if word not in self.stopwords]
        log.info('Stripped stop words: %s' % ((set(words) - set(filtered)) or '-'))
        return filtered

    def search(self, words):
        pass

    def scrape(self, url):
        """
        Scrape a web page using prerender.

        :param url: URL
        :return: text contents
        """

        page = requests.get('http://{host}:{port}/{url}'.format(
            host=self.prerender_host,
            port=self.prerender_port,
            url=url))
        soup = BeautifulSoup(page.text, 'lxml')
        [s.extract() for s in soup(['iframe', 'script', 'style'])]

        page_content = soup.get_text()
        if page_content:
            log.info('Scraped {len} characters from URL: {url}'.format(len=len(page_content), url=url))
        else:
            log.warning('Unable to scrape any content for URL: %s' % url)

        return page_content

    def scrape_contents(self, documents):
        log.info('Scraping results')

        for doc in documents:
            text_content = None
            if self.scrape_cache:
                try:
                    cached_content = self.scrape_cache.get(doc['url'])
                    if cached_content:
                        text_content = str(cached_content)
                        log.info(
                            'Found page content (%s chars) from scrape cache: %s' % (len(text_content), doc['url']))
                except TypeError:
                    pass

            if not text_content:
                log.debug('Scraping document %s:  %s' % (doc['name'], doc['url']))

                text_content = self.scrape(doc['url'])
                if self.scrape_cache and text_content:
                    log.info('Adding page to scrape cache: %s' % (doc['url']))
                    self.scrape_cache.setex(doc['url'], self.scrape_cache_expire, text_content)

            if text_content:
                doc['contents'] = text_content

        return documents

    def topic_model(self, documents):
        log.info('Topic modeling')

        vectorizer = CountVectorizer(stop_words=list(self.stopwords))
        data_corpus = [r.get('contents', '') for r in documents]

        if len(documents) < 3 or not any(data_corpus):
            log.error('Not enough documents for topic modeling, or corpus empty.')
            return documents, [[]]

        X = vectorizer.fit_transform(data_corpus)
        vocab = vectorizer.get_feature_names()

        # TODO: This could be approximated using path length of terms in KOKO
        n_topics = 1 + min(round(2 * math.sqrt(len(documents))), 9)

        model = lda.LDA(n_topics=n_topics, n_iter=800, random_state=1)
        model.fit(X)
        topic_word = model.topic_word_
        n_top_words = 10
        topics_words = []

        eps = 0.001
        for i, topic_dist in enumerate(topic_word):
            wordindex = np.argsort(topic_dist)[::-1]  # rev sort
            weights = topic_dist[wordindex]  # Topic word weights
            words = [np.array(vocab)[wordindex[j]] for j in range(min(n_top_words, len(wordindex))) if weights[j] > eps]
            word_weights = [weights[j] for j in range(min(n_top_words, len(wordindex))) if weights[j] > eps]
            log.info('Topic {}: {}; {}'.format(i, ', '.join(words), ', '.join(map(str, word_weights))))
            topics_words.append(tuple(zip(words, word_weights)))

        doc_topics = model.doc_topic_

        for (doc, topic) in zip(documents, doc_topics):
            doc['topic'] = topic.tolist()

        return documents, topics_words


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


class RFSearch_GoogleAPI(RFSearch):
    """
    Relevance-feedback search using Google Custom Search.
    """

    def __init__(self,
                 apikey='',
                 arpa_url='http://demo.seco.tkk.fi/arpa/koko-related',
                 stopwords=None,
                 scrape_cache=None,
                 **kwargs):

        super().__init__(stopwords=stopwords, scrape_cache=scrape_cache, **kwargs)
        self.search_service = build("customsearch", "v1", developerKey=apikey)
        if arpa_url:
            self.word_expander = SearchExpanderArpa(arpa_url=arpa_url).expand_words
        else:
            self.word_expander = lambda words: [(word,) for word in words]

    @staticmethod
    def combine_expanded(expanded):
        return [' OR '.join(wordset) for wordset in expanded]

    def search(self, words, expand_words=True):
        """
        Create a search query based on a list of words and query for results.

        :param words: list of words
        :param expand_words:
        :return:
        """
        if expand_words:
            words = self.combine_expanded(self.word_expander(words))

        query = ' '.join(words)
        while len(query) > 2500:
            words.pop()
            query = ' '.join(words)

        log.info('Query: %s' % query)

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
            sanitized.append({'name': item['title'], 'url': item['link'], 'description': item['snippet']})

        log.info('Returning %s results from search.' % len(sanitized))

        return sanitized


class RFSearch_GoogleUI(RFSearch):
    """
    Relevance-feedback search using Google Custom Search.
    """

    def __init__(self, arpa_url='http://demo.seco.tkk.fi/arpa/koko-related',
            stopwords=None, scrape_cache=None, **kwargs):
        super().__init__(stopwords=stopwords, scrape_cache=scrape_cache, **kwargs)
        self.num_results = 10
        if arpa_url:
            self.word_expander = SearchExpanderArpa(arpa_url=arpa_url).expand_words
        else:
            self.word_expander = lambda words: [(word,) for word in words]

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

        return sanitized


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("--engine", help="Search engine", default='GoogleAPI', choices=["GoogleAPI", "GoogleUI"])
    argparser.add_argument("--apikey", help="Google API key")
    argparser.add_argument('words', metavar='Keywords', type=str, nargs='+', help='search keywords')
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = argparser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    engines = {'GoogleAPI': RFSearch_GoogleAPI,
               'GoogleUI': RFSearch_GoogleUI,
               }

    apikey = args.apikey
    search_words = args.words
    engine = args.engine

    np.set_printoptions(precision=3, suppress=True)

    with open('stopwords.txt', 'r') as f:
        stopwords = f.read().split()

    params = {}
    params.update({'stopwords': stopwords})
    if apikey:
        params.update({'apikey': apikey})
    searcher = engines[engine](**params)

    docs = searcher.search(search_words)
    # docs = pickle.load(open('google_search_results.pkl', 'rb'))
    # pickle.dump(res, open('google_search_results.pkl', 'wb'))

    docs = searcher.scrape_contents(docs)

    docs, _ = searcher.topic_model(docs)

    for doc in docs:
        print(doc['name'].upper())
        print(doc['description'])
        print(doc.get('topic'))
        print()
