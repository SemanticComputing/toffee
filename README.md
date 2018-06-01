# Toffee -- Topical Relevance Feedback Search

## Requirements

- Hades for news corpus index in elasticsearch: https://github.com/SemanticComputing/hades
    - Elasticsearch uses the Hades `esdata` volume directly if present

## Running the initial training using the news corpus

```
docker-compose -f docker-compose-train.yml up
```

## Running locally

Add a file named `.env` to the repository root with `API_KEY=` and the Google search API key.

```
docker-compose build --build-arg 'REACT_APP_BACKEND=http://localhost:5000' frontend
docker-compose up -d
```
