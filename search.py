#!/usr/bin/env python3
#  -*- coding: UTF-8 -*-
"""
Relevance feedback search using semantic knowledge and topic modeling.
"""
import argparse
import pprint

from arpa_linker.arpa import post
from googleapiclient.discovery import build


def expand_words(words, url='http://demo.seco.tkk.fi/arpa/koko-related'):
    expanded = []
    for word in words:
        data = {'text': word}
        query_result = post(url, data, retries=5)

        related = [match['properties'].get('relatedlabel', '') for match in query_result['results']]
        related = [item for sublist in related for item in set(sublist)]
        related = [x if ' ' in x else x.strip('"') for x in related]

        # print('Related concepts: %s' % (len(related)))
        expanded.append(tuple(set(related + [word])))

    return expanded

argparser = argparse.ArgumentParser(description="Process war prisoners CSV", fromfile_prefix_chars='@')
argparser.add_argument("apikey", help="Google API key")
argparser.add_argument('words', metavar='N', type=str, nargs='+', help='search words')
args = argparser.parse_args()

apikey = args.apikey
search_words = args.words

print('Querying ARPA')
expanded_words = expand_words(search_words)
print('Expanded from %s words to %s words' % (len(search_words), len([x for y in expanded_words for x in y])))
query = ' '.join([' OR '.join(wordset) for wordset in expanded_words])
print(query)

service = build("customsearch", "v1", developerKey=apikey)

res = service.cse().list(
    q=' '.join(search_words),
    cx='012121639191539030590:cshq4wzc7ms',
    # excludeTerms='foo'
).execute()
items = res.get('items')
pprint.pprint(items)
