#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search using semantic knowledge and topic modeling.
"""
import argparse

import pickle
from arpa_linker.arpa import post
from google import google
from googleapiclient.discovery import build

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import lda


class RFSearch:
    """
    Relevance-feedback search abstract class.
    """

    def __init__(self):
        pass

    def search(self, words):
        pass


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
        expanded = []
        for word in words:
            data = {'text': word}
            query_result = post(self.arpa_url, data, retries=5, wait=5)

            related = [match['properties'].get('relatedlabel', '') for match in query_result['results']]
            related = [item for sublist in related for item in set(sublist)]
            related = [x if ' ' in x else x.strip('"') for x in related]

            expanded.append(tuple(set(related + [word])))

        return expanded


class RFSearch_GoogleAPI(RFSearch, SearchExpanderArpa):
    """
    Relevance-feedback search using Google Custom Search.
    """

    def __init__(self, apikey='', arpa_url='http://demo.seco.tkk.fi/arpa/koko-related'):
        RFSearch.__init__(self)
        SearchExpanderArpa.__init__(self, arpa_url=arpa_url)
        self.search_service = build("customsearch", "v1", developerKey=apikey)

    def search(self, words):
        """
        Create a search query based on a list of words and query for results.

        :param words:
        :return:
        """
        print('Querying ARPA')
        expanded_words = self.expand_words(words)
        print('Expanded from %s words to %s words' % (len(words), len([x for y in expanded_words for x in y])))
        query = ' '.join([' OR '.join(wordset) for wordset in expanded_words])
        print('Query:')
        print(query)

        res = self.search_service.cse().list(
            q=query,
            cx='012121639191539030590:cshq4wzc7ms',
            num=10,
            start=1,
            # excludeTerms='foo'
        ).execute()

        items = res.get('items')
        next_page = True  # res.get('nextPage')
        i = 1

        total_results = int(res.get('searchInformation').get('totalResults'))
        if items:
            print('Got %s results' % total_results)
            # pprint.pprint(items)
        #     pickle.dump(res, open('results.pkl', 'wb'))
        #     print('Results saved to file.')
        # else:
        #     print(res)

        while total_results > 10 and next_page and i < 50:
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

        sanitized = []
        for item in items:
            sanitized.append({'name': item['title'], 'url': item['link'], 'description': item['snippet']})

        return sanitized


class RFSearch_GoogleUI(RFSearch, SearchExpanderArpa):
    """
    Relevance-feedback search using Google Custom Search.
    """

    def __init__(self, arpa_url='http://demo.seco.tkk.fi/arpa/koko-related'):
        RFSearch.__init__(self)
        SearchExpanderArpa.__init__(self, arpa_url=arpa_url)
        self.num_results = 10

    def search(self, words):
        """
        Create a search query based on a list of words and query for results.

        :param words:
        :return:
        """
        print('Querying ARPA')
        expanded_words = self.expand_words(words)
        print('Expanded from %s words to %s words' % (len(words), len([x for y in expanded_words for x in y])))
        query = ' '.join([' OR '.join(wordset) for wordset in expanded_words])
        print('Query:')
        print(query)

        res = google.search(query, self.num_results)

        sanitized = []
        for item in res:
            sanitized.append({'name': item.name, 'url': item.link, 'description': item.description})

        return sanitized


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("--engine", help="Search engine", default='GoogleAPI', choices=["GoogleAPI", "GoogleUI"])
    argparser.add_argument("--apikey", help="Google API key")
    argparser.add_argument('words', metavar='Keywords', type=str, nargs='+', help='search keywords')
    args = argparser.parse_args()

    engines = {'GoogleAPI': RFSearch_GoogleAPI,
               'GoogleUI': RFSearch_GoogleUI,
               }

    apikey = args.apikey
    search_words = args.words
    engine = args.engine

    params = {}
    if apikey:
        params.update({'apikey': apikey})
    searcher = engines[engine](**params)

    with open('stopwords.txt', 'r') as f:
        stopwords = f.read().split()

    res = searcher.search(search_words)
    # res = pickle.load(open('google_search_results.pkl', 'rb'))
    # pickle.dump(res, open('google_search_results.pkl', 'wb'))

    print(len(res))
    print(res[0])
    print()

    for r in res:
        print('%s:  %s' % (r['name'], r['url']))

    # TODO: Use dryscrape to get page contents

    vectorizer = CountVectorizer(stop_words=stopwords)
    data_corpus = (r['description'] for r in res)
    X = vectorizer.fit_transform(data_corpus)
    vocab = vectorizer.get_feature_names()

    print(vocab)
    print(X.toarray())

    model = lda.LDA(n_topics=len(res) // 2, n_iter=1500, random_state=1)
    model.fit(X)
    topic_word = model.topic_word_
    n_top_words = 8

    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

