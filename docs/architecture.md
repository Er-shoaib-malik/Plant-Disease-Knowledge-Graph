# Architecture

1. Next.js accepts and previews an image, then calls FastAPI.
2. FastAPI validates MIME, byte size, image decoding, and dimensions.
3. A process-wide `Predictor` loads the trained ResNet-50 checkpoint once and returns top-k model scores.
4. The predicted class is normalized and matched to a Neo4j disease record. Graph retrieval returns only disease-linked crop, pathogen, symptoms, conditions, treatments, prevention, and source nodes.
5. Ollama or the configured OpenAI API receives prediction metadata and that retrieved context. If it is unavailable or malformed, the API returns retrieved facts with an explicit fallback status.
6. FastAPI returns a Pydantic validated report. The frontend renders sources and limitations and offers print-to-PDF.

Neo4j and the trained checkpoint are optional at startup; unavailable state is reflected by health endpoint and diagnosis response. No disease prediction is synthesized when the checkpoint is missing.
