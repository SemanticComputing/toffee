# Toffee -- Topical Relevance Feedback Search

## Running locally

Add a file named `.env` to the repository root with `API_KEY=` and the Google search API key.

```
docker-compose build --build-arg 'REACT_APP_BACKEND=http://localhost:5000' frontend
docker-compose up -d
```
