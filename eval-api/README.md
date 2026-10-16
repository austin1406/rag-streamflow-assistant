# Streamflow Eval API

A small Spring Boot REST API, part of the [Streamflow RAG Assistant](../README.md), that stores
and summarizes its evaluation runs. Each run records which evaluation questions retrieval got
right and, when a generation backend was available, which answers were correct, then reports hit
rates so you can compare runs over time. The RAG project publishes results to it with
`python -m eval.publish_results` (see the root README).

**Stack:** Java 21, Spring Boot 4, Spring Data JPA, Bean Validation, H2, JUnit 5, Mockito, MockMvc.

## Run

```bash
./mvnw spring-boot:run          # http://localhost:8080
./mvnw test
```

Data lives in memory by default. To keep it between restarts:
`DB_URL=jdbc:h2:file:./data/evalapi ./mvnw spring-boot:run`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/runs` | Record a run (201 + `Location`) |
| GET | `/api/runs` | List run summaries, newest first |
| GET | `/api/runs/{id}` | One run with per-question results |
| DELETE | `/api/runs/{id}` | Delete a run and its results (204) |

```bash
curl -X POST localhost:8080/api/runs -H "Content-Type: application/json" -d '{
  "label": "baseline", "model": "llama3.2:1b", "topK": 5,
  "results": [
    {"question": "What R-squared did the model get on Bunchgrass Meadow?", "retrievalOk": true, "answerOk": true},
    {"question": "Who introduced DACP?", "retrievalOk": false, "answerOk": null}
  ]}'
```

`answerOk` is optional: it is `null` when no generation backend was available. Summaries report
`answerRate` as `null` in that case instead of a misleading 0%, and only graded answers count
toward it. Invalid requests return a 400 `ProblemDetail` with a per-field `errors` map; missing
runs return 404.

## Tests (26)

- `RunSummaryTest`: rate math, null handling, rounding, empty runs.
- `EvalRunServiceTest`: Mockito unit tests for create/list/get/delete, trimming, and fixed-clock timestamps.
- `EvalRunRepositoryTest`: `@DataJpaTest` against H2 (ordering, cascade delete, eager loading).
- `EvalRunControllerTest`: `@WebMvcTest` for status codes, validation errors, and malformed JSON.
- `EvalRunApiIntegrationTest`: full-stack create, read, list, delete through MockMvc and H2.

CI runs them whenever `eval-api/` changes (`.github/workflows/eval-api.yml` at the repo root).
