# API

Interactive OpenAPI docs are served at `http://localhost:8000/docs`.

* `GET /api/v1/health` reports API, model, Neo4j and LLM configuration status.
* `POST /api/v1/predict` accepts multipart field `file`; returns ranked predictions and an uncalibrated confidence marker.
* `POST /api/v1/diagnose` accepts `file`; returns a structured report, evidence status and source list.
* `GET /api/v1/diseases` lists graph records; `GET /api/v1/diseases/{disease_id}` returns disease context.
* `GET /api/v1/graph/{disease_id}` returns the same bounded linked subgraph as disease detail.
* `POST /api/v1/feedback` accepts `{ "agrees": bool, "corrected_disease": string|null, "comment": string|null }`; appends non-identifying submissions to `FEEDBACK_PATH` (default `data/feedback.jsonl`). Set a retention policy before collecting it in a deployed service.

Image endpoints accept image MIME types up to `MAX_UPLOAD_MB`; decompressed images over 25 megapixels are rejected.
