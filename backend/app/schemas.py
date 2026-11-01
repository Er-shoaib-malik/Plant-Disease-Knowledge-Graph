from pydantic import BaseModel, Field

class Prediction(BaseModel):
    class_name: str
    confidence: float = Field(ge=0,le=1)

class Source(BaseModel):
    title: str
    url: str

class Diagnosis(BaseModel):
    diagnosis: str
    crop: str
    disease: str
    confidence: float = Field(ge=0,le=1)
    uncertain: bool = False
    pathogen: str | None = None
    symptoms: list[str] = []
    environmental_factors: list[str] = []
    severity_assessment: str
    treatment_recommendations: list[str] = []
    prevention: list[str] = []
    limitations: list[str] = []
    sources: list[Source] = []
    top_predictions: list[Prediction] = []
    graph_available: bool = False
    llm_available: bool = False

class Feedback(BaseModel):
    prediction_id: str | None = None
    agrees: bool
    corrected_disease: str | None = None
    comment: str | None = Field(default=None,max_length=1000)
