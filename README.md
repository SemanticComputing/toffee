# Toffee -- Topical Relevance Feedback Search

Iterative relevance feedback search using topic modeling and user feedback to guide the search towards the topics of interest. 

More details can be found in:
> Mikko Koho, Erkki Heino, Arttu Oksanen and Eero Hyvönen: Toffee - Semantic Media Search Using Topic Modeling and Relevance Feedback. Proceedings of the ISWC 2018 Posters & Demonstrations, Industry and Blue Sky Ideas Tracks, CEUR Workshop Proceedings, Monterey, California, USA, October, 2018. Vol 2180. [Pre-print PDF](https://seco.cs.aalto.fi/publications/2018/koho-et-al-toffee-demo-2018.pdf)

## Requirements

- Hades for news corpus index in elasticsearch: https://github.com/SemanticComputing/hades
    - Elasticsearch uses the Hades `esdata` volume directly if present

## Architecture

![Toffee system architecture](toffee_system.png)

## Running the initial training using the news corpus

```
docker-compose -f docker-compose-train.yml up
```

## Running locally

Add a file named `.env` to the repository root with `API_KEY=` and the Google search API key. The `REACT_APP_BACKEND` should point to the `web` service address at the hosting server.

```
docker-compose build --build-arg 'REACT_APP_BACKEND=http://localhost:5000' frontend
docker-compose up -d
```

To deploy with several worker and prerender instances: (for docker version < 3.0)

```
docker-compose build --build-arg 'REACT_APP_BACKEND=http://localhost:5000' frontend
docker-compose up -d
docker-compose scale worker=3 prerender=3
```

 
