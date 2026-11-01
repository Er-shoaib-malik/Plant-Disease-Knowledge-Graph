from backend.app.config import settings

class GraphStore:
    def __init__(self, uri=None, user=None, password=None):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(uri or settings.neo4j_uri, auth=(user or settings.neo4j_user, password or settings.neo4j_password))
    def close(self): self.driver.close()
    def health(self):
        try:
            self.driver.verify_connectivity(); return True
        except Exception: return False
    def retrieve(self, disease_id):
        query = """MATCH (d:Disease {id:$id}) OPTIONAL MATCH (c:Crop)-[:SUSCEPTIBLE_TO]->(d) OPTIONAL MATCH (d)-[:CAUSED_BY]->(p:Pathogen) OPTIONAL MATCH (d)-[:HAS_SYMPTOM]->(s:Symptom) OPTIONAL MATCH (d)-[:FAVORED_BY]->(e:EnvironmentalCondition) OPTIONAL MATCH (d)-[:TREATED_BY]->(t:Treatment) OPTIONAL MATCH (d)-[:PREVENTED_BY]->(v:Prevention) OPTIONAL MATCH (d)-[:SUPPORTED_BY]->(src:AgriculturalSource) RETURN d,c,p,collect(DISTINCT s) AS symptoms,collect(DISTINCT e) AS environment,collect(DISTINCT t) AS treatments,collect(DISTINCT v) AS prevention,collect(DISTINCT src) AS sources"""
        with self.driver.session() as session:
            row=session.run(query,id=disease_id).single()
        if not row: return None
        node=lambda n: dict(n) if n else None
        return {"disease":node(row["d"]),"crop":node(row["c"]),"pathogen":node(row["p"]),"symptoms":[node(n) for n in row["symptoms"] if n],"environmental_factors":[node(n) for n in row["environment"] if n],"treatments":[node(n) for n in row["treatments"] if n],"prevention":[node(n) for n in row["prevention"] if n],"sources":[node(n) for n in row["sources"] if n]}
    def list_diseases(self):
        with self.driver.session() as session:
            return [dict(r["d"])|{"crop":dict(r["c"]) if r["c"] else None} for r in session.run("MATCH (d:Disease) OPTIONAL MATCH (c:Crop)-[:SUSCEPTIBLE_TO]->(d) RETURN d,c ORDER BY d.name")]
