import sys
from types import SimpleNamespace
from graph.connection import GraphStore
from backend.app.schemas import Diagnosis

class FakeSession:
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def run(self,query,**params):
        self.query=query; self.params=params
        return SimpleNamespace(single=lambda:{"d":{"id":"tomato_early_blight","name":"Early Blight"},"c":{"name":"Tomato"},"p":None,"symptoms":[],"environment":[],"treatments":[],"prevention":[],"sources":[]})

class FakeDriver:
    def __init__(self): self.s=FakeSession()
    def session(self): return self.s

def test_graph_retrieval_uses_parameterized_id(monkeypatch):
    driver=FakeDriver()
    fake=SimpleNamespace(GraphDatabase=SimpleNamespace(driver=lambda *args,**kwargs:driver))
    monkeypatch.setitem(sys.modules,"neo4j",fake)
    store=GraphStore()
    result=store.retrieve("tomato_early_blight")
    assert result["disease"]["name"]=="Early Blight"
    assert driver.s.params=={"id":"tomato_early_blight"}
    assert "$id" in driver.s.query

def test_diagnosis_schema_requires_core_report_fields():
    record=Diagnosis(diagnosis="Check source.",crop="Tomato",disease="Early Blight",confidence=.6,severity_assessment="Unknown")
    assert record.uncertain is False
    assert record.sources==[]
