from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.main import resolve

def test_plantvillage_name_mapping():
    assert resolve("Tomato___Early_blight")==("Tomato","Early Blight","tomato_early_blight")
    assert resolve("Apple___Apple_scab")==("Apple","Scab","apple_scab")
    assert resolve("Corn_(maize)___Common_rust_")==("Corn","Common Rust","corn_common_rust")

def test_health_reports_missing_model(monkeypatch):
    import backend.app.main as main
    monkeypatch.setattr(main,"predictor",type("P",(),{"available":False,"error":"missing checkpoint"})())
    response=TestClient(app).get("/api/v1/health")
    assert response.status_code==200
    assert response.json()["model"]["available"] is False

def test_invalid_upload_type():
    response=TestClient(app).post("/api/v1/predict",files={"file":("x.txt",b"text","text/plain")})
    assert response.status_code==415

def test_missing_model_does_not_fake_prediction(monkeypatch):
    import io
    import backend.app.main as main
    monkeypatch.setattr(main,"predictor",type("P",(),{"available":False,"error":"missing checkpoint"})())
    from PIL import Image
    buf=io.BytesIO(); Image.new("RGB",(8,8),(20,120,40)).save(buf,format="PNG")
    response=TestClient(app).post("/api/v1/predict",files={"file":("leaf.png",buf.getvalue(),"image/png")})
    assert response.status_code==503

def test_feedback_is_saved_without_identity(monkeypatch,tmp_path):
    import backend.app.main as main
    monkeypatch.setattr(main.settings,"feedback_path",str(tmp_path/"feedback.jsonl"))
    response=TestClient(app).post("/api/v1/feedback",json={"agrees":False,"corrected_disease":"Early blight","comment":"Needs review"})
    assert response.status_code==202
    assert response.json()["stored"] is True
    saved=(tmp_path/"feedback.jsonl").read_text()
    assert '"corrected_disease": "Early blight"' in saved
    assert '"user"' not in saved
