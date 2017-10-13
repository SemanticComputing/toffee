#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search using semantic knowledge and topic modeling.
"""
import argparse
import pprint

import pickle
from arpa_linker.arpa import post
from googleapiclient.discovery import build
from google import google


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

    def __init__(self, apikey, arpa_url='http://demo.seco.tkk.fi/arpa/koko-related'):
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
            # excludeTerms='foo'
        ).execute()

        return res


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

        return res

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Search CLI", fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument('words', metavar='N', type=str, nargs='+', help='search words')
    args = argparser.parse_args()

    apikey = args.apikey
    search_words = args.words

    searcher = RFSearch_GoogleAPI(apikey)

    res = searcher.search(search_words)
    items = res.get('items')
    if items:
        print('Got %s results' % res.get('searchInformation').get('totalResults'))
        # pprint.pprint(items)
        pickle.dump(res, open('results.pkl', 'wb'))
        print('Results saved to file.')
    else:
        print(res)
