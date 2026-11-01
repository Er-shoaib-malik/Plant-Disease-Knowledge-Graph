import json
import httpx
from .config import settings

SYSTEM = """You are an agricultural disease explanation assistant. Explain the image classification using ONLY the supplied model results and retrieved agricultural facts. Do not invent scientific facts, causes, symptoms, treatment, dosages, or safety instructions. Distinguish prediction from retrieved facts and state what remains unknown. Never recommend chemicals unless crop suitability, usage, safety conditions and source are verified. Image prediction is not certainty. Return only JSON with diagnosis, crop, disease, confidence, pathogen, symptoms, environmental_factors, severity_assessment, treatment_recommendations, prevention, limitations, sources. Use simple language."""

async def explain(context):
    try:
        if settings.llm_provider == "openai" and settings.openai_api_key:
            async with httpx.AsyncClient(timeout=30) as client:
                r=await client.post("https://api.openai.com/v1/chat/completions",headers={"Authorization":f"Bearer {settings.openai_api_key}"},json={"model":settings.openai_model,"response_format":{"type":"json_object"},"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(context)}]})
                r.raise_for_status(); return json.loads(r.json()["choices"][0]["message"]["content"])
        async with httpx.AsyncClient(timeout=30) as client:
            r=await client.post(f"{settings.ollama_base_url}/api/chat",json={"model":settings.ollama_model,"stream":False,"format":"json","messages":[{"role":"system","content":SYSTEM},{"role":"user","content":json.dumps(context)}]})
            r.raise_for_status(); return json.loads(r.json()["message"]["content"])
    except Exception:
        return None
