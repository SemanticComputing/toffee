#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse

from flask import Flask, request, json
from flask_cors import CORS

from search import RFSearch_GoogleAPI

app = Flask(__name__)
CORS(app)


@app.route('/')
def hello():
    return __doc__


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.values.get('query')
    print('Search API got words: %s' % query)
    if query:
        res = searcher.search(query.split())
        print('Got %s results' % res.get('searchInformation').get('totalResults'))
        items = res.get('items')
        return json.dumps(items)

    return ''


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument("--host", default=None, help="Host (e.g. 0.0.0.0)")
    args = argparser.parse_args()

    searcher = RFSearch_GoogleAPI(args.apikey)

    app.run(host=args.host)
