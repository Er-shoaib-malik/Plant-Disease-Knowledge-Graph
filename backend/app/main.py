import io, json, logging, re
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from .config import settings
from .schemas import Diagnosis, Feedback
from .explainer import explain
from graph.connection import GraphStore
from ml.inference import Predictor

logging.basicConfig(level=logging.INFO); log=logging.getLogger("plant-disease")
predictor=None; graph=None
@asynccontextmanager
async def lifespan(app):
    global predictor,graph
    predictor=Predictor(settings.model_checkpoint)
    try: graph=GraphStore()
    except Exception as e: log.warning("Graph connection unavailable: %s",e); graph=None
    yield
    if graph:
        try: graph.close()
        except Exception: pass

app=FastAPI(title="Plant Disease Intelligence API",version="1.0.0",description="Image inference with sourced agricultural graph retrieval.",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=[settings.frontend_origin,"http://localhost:3000"],allow_credentials=False,allow_methods=["GET","POST"],allow_headers=["*"])

async def read_image(file):
    if not file.content_type or not file.content_type.startswith("image/"): raise HTTPException(415,"Upload a supported image file.")
    data=await file.read(settings.max_upload_mb*1024*1024+1)
    if len(data)>settings.max_upload_mb*1024*1024: raise HTTPException(413,f"Image exceeds {settings.max_upload_mb} MB limit.")
    try:
        source=Image.open(io.BytesIO(data))
        if source.width*source.height>25_000_000: raise HTTPException(413,"Image dimensions are too large.")
        source.verify(); return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError,OSError): raise HTTPException(400,"The uploaded file is not a valid image.")

def resolve(name):
    parts=re.split(r"___|__+",name.lower().strip("_"),maxsplit=1)
    crop_raw=" ".join(parts[0].replace("_"," ").replace("("," ").replace(")"," ").split())
    disease_raw=parts[1] if len(parts)>1 else ""
    crop_map={"tomato":"Tomato","potato":"Potato","apple":"Apple","grape":"Grape","corn":"Corn","corn maize":"Corn","maize":"Corn","pepper":"Pepper"}
    crop=crop_map.get(crop_raw,crop_raw.title()); disease=disease_raw.replace("_"," ").strip().title()
    if disease.lower().startswith(crop.lower()+" "): disease=disease[len(crop)+1:]
    return crop,disease,re.sub(r"[^a-z0-9]+","_",f"{crop}_{disease}".lower()).strip("_")

async def predictions(file):
    image=await read_image(file)
    if not predictor or not predictor.available: raise HTTPException(503,predictor.error if predictor else "Inference model is unavailable")
    try: return image,predictor.predict(image)
    except Exception as e: log.exception("Inference failed"); raise HTTPException(500,"Image inference failed.") from e

@app.get("/api/v1/health")
async def health():
    llm_available=False
    if settings.llm_provider=="ollama":
        try:
            import httpx
            async with httpx.AsyncClient(timeout=2) as client:
                response=await client.get(f"{settings.ollama_base_url}/api/tags"); llm_available=response.is_success
        except Exception: pass
    elif settings.llm_provider=="openai": llm_available=bool(settings.openai_api_key)
    return {"api":"ok","model":{"available":bool(predictor and predictor.available),"error":predictor.error if predictor and not predictor.available else None},"database":{"available":bool(graph and graph.health())},"llm":{"available":llm_available,"configured":settings.llm_provider=="ollama" or bool(settings.openai_api_key),"provider":settings.llm_provider}}

@app.post("/api/v1/predict")
async def predict(file: UploadFile=File(...)):
    _,top=await predictions(file)
    return {"top_predictions":top,"uncertain":top[0]["confidence"]<settings.low_confidence_threshold,"confidence_is_calibrated":False}

@app.post("/api/v1/diagnose",response_model=Diagnosis)
async def diagnose(file: UploadFile=File(...)):
    image,top=await predictions(file); crop,disease,did=resolve(top[0]["class_name"]); facts=None; graph_ok=False
    if graph:
        try: facts=graph.retrieve(did); graph_ok=facts is not None
        except Exception as e: log.warning("Graph retrieval failed: %s",e)
    # Seed records may have no relationships yet; resolve by normalized stored name if model spelling differs.
    if facts is None and graph:
        try:
            for row in graph.list_diseases():
                if str(row.get("name","")).lower()==disease.lower() and str((row.get("crop") or {}).get("name","")).lower()==crop.lower():
                    facts=graph.retrieve(row["id"]); graph_ok=facts is not None; break
        except Exception: pass
    if facts is None:
        return Diagnosis(diagnosis="The image model prediction requires verification; no matching graph evidence is available.",crop=crop,disease=disease,confidence=top[0]["confidence"],uncertain=top[0]["confidence"]<settings.low_confidence_threshold,pathogen=None,symptoms=[],environmental_factors=[],severity_assessment="Cannot assess severity from this image classification alone.",treatment_recommendations=[],prevention=[],limitations=["Knowledge Graph unavailable or no matching sourced disease record was found.","Image classification is not a confirmed diagnosis."],sources=[],top_predictions=top,graph_available=False,llm_available=False)
    context={"crop":crop,"disease":disease,"confidence":top[0]["confidence"],"top_predictions":top,"retrieved_facts":facts}
    answer=await explain(context)
    if answer:
        try:
            # Do not let model-generated unsupported facts leak into result; use retrieved fact fields only.
            answer.update({"crop":crop,"disease":disease,"confidence":top[0]["confidence"],"uncertain":top[0]["confidence"]<settings.low_confidence_threshold,"pathogen":(facts.get("pathogen") or {}).get("name"),"symptoms":[x.get("name","") for x in facts["symptoms"]],"environmental_factors":[x.get("name","") for x in facts["environmental_factors"]],"treatment_recommendations":[x.get("name","") for x in facts["treatments"]],"prevention":[x.get("name","") for x in facts["prevention"]],"sources":[{"title":x.get("name",x.get("title","Source")),"url":x.get("url","")} for x in facts["sources"]],"top_predictions":top,"graph_available":True,"llm_available":True})
            return Diagnosis(**answer)
        except Exception as e: log.warning("LLM output validation failed; using retrieved facts: %s",e)
    d=facts.get("disease") or {}; src=facts.get("sources",[])
    return Diagnosis(diagnosis=d.get("description") or "Retrieved facts are limited; consult the cited source for diagnosis and management.",crop=crop,disease=disease,confidence=top[0]["confidence"],uncertain=top[0]["confidence"]<settings.low_confidence_threshold,pathogen=(facts.get("pathogen") or {}).get("name"),symptoms=[x.get("name","") for x in facts["symptoms"]],environmental_factors=[x.get("name","") for x in facts["environmental_factors"]],severity_assessment="Severity cannot be determined from classification confidence.",treatment_recommendations=[x.get("name","") for x in facts["treatments"]],prevention=[x.get("name","") for x in facts["prevention"]],limitations=["LLM unavailable; this report presents retrieved graph facts directly.","Image classification is not a confirmed diagnosis."],sources=[{"title":x.get("name",x.get("title","Source")),"url":x.get("url","")} for x in src],top_predictions=top,graph_available=True,llm_available=False)

@app.get("/api/v1/diseases")
def diseases():
    if not graph or not graph.health(): raise HTTPException(503,"Knowledge Graph is unavailable")
    return graph.list_diseases()

@app.get("/api/v1/diseases/{disease_id}")
def disease(disease_id: str):
    if not graph: raise HTTPException(503,"Knowledge Graph is unavailable")
    result=graph.retrieve(disease_id)
    if not result: raise HTTPException(404,"Disease not found")
    return result

@app.get("/api/v1/graph/{disease_id}")
def graph_data(disease_id: str): return disease(disease_id)

@app.post("/api/v1/feedback",status_code=202)
def feedback(payload: Feedback):
    path=Path(settings.feedback_path)
    try:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps({**payload.model_dump(),"submitted_at":datetime.now(timezone.utc).isoformat()})+"\n")
    except OSError:
        log.exception("Feedback persistence failed")
        raise HTTPException(503,"Feedback could not be stored on this server.")
    return {"accepted":True,"stored":True,"message":"Feedback stored without user identity."}
