import json
from pathlib import Path
from backend.app.config import settings

def ingest(uri=None,user=None,password=None,seed_path=Path(__file__).with_name("seed.json")):
    from neo4j import GraphDatabase
    rows=json.loads(Path(seed_path).read_text()); driver=GraphDatabase.driver(uri or settings.neo4j_uri,auth=(user or settings.neo4j_user,password or settings.neo4j_password))
    with driver.session() as s:
        s.run("CREATE CONSTRAINT disease_id IF NOT EXISTS FOR (n:Disease) REQUIRE n.id IS UNIQUE")
        s.run("CREATE CONSTRAINT crop_id IF NOT EXISTS FOR (n:Crop) REQUIRE n.id IS UNIQUE")
        for r in rows:
            s.run("""MERGE (c:Crop {id:toLower($crop)}) SET c.name=$crop MERGE (d:Disease {id:$id}) SET d.name=$disease,d.description=$description MERGE (c)-[:SUSCEPTIBLE_TO]->(d) WITH d,$r AS r FOREACH (_ IN CASE WHEN r.source_url='' THEN [] ELSE [1] END | MERGE (src:AgriculturalSource {id:r.source_url}) SET src.name=r.source_title,src.url=r.source_url MERGE (d)-[:SUPPORTED_BY]->(src)) FOREACH (x IN r.symptoms | MERGE (n:Symptom {id:toLower(replace(x,' ','_'))}) SET n.name=x MERGE (d)-[:HAS_SYMPTOM]->(n)) FOREACH (x IN r.environmental_factors | MERGE (n:EnvironmentalCondition {id:toLower(replace(x,' ','_'))}) SET n.name=x MERGE (d)-[:FAVORED_BY]->(n)) FOREACH (_ IN CASE WHEN r.pathogen='' THEN [] ELSE [1] END | MERGE (p:Pathogen {id:toLower(replace(r.pathogen,' ','_'))}) SET p.name=r.pathogen MERGE (d)-[:CAUSED_BY]->(p))""",r=r,**r)
    driver.close(); return len(rows)

if __name__=="__main__": print(f"Ingested {ingest()} sourced disease records")
