#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Relevance feedback search API.
"""
import argparse
import logging

from flask import Flask, request, json
from flask_cors import CORS

from search import RFSearch_GoogleAPI

app = Flask(__name__)
CORS(app)

log = logging.getLogger(__name__)


@app.route('/')
def hello():
    return __doc__


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.values.get('query')
    log.info('Search API got words: %s' % query)
    if query:
        items = searcher.search(query.split())
        log.info('Got %s results' % len(items))
        return json.dumps(items)

    return ''


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__, fromfile_prefix_chars='@')
    argparser.add_argument("apikey", help="Google API key")
    argparser.add_argument("--host", default=None, help="Host (e.g. 0.0.0.0)")
    argparser.add_argument("--loglevel", default='INFO', help="Logging level, default is INFO.",
                           choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    args = argparser.parse_args()

    logging.basicConfig(level=getattr(logging, args.loglevel),
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    searcher = RFSearch_GoogleAPI(args.apikey)

    app.run(host=args.host)
