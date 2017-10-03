#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse
import pprint

from flask import Flask, request, json

from search import RFSearch

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

@app.route('/search')
def search():
    words = request.args.get('words', default='', type=str)
    print('Search API got words: %s' % words)
    res = searcher.search(words.split())
    print('Got %s results' % res.get('searchInformation').get('totalResults'))
    items = res.get('items')
    return json.dumps(items)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Process war prisoners CSV", fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    args = argparser.parse_args()

    searcher = RFSearch(args.apikey)

    app.run()
