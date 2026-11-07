import tempfile,unittest
from pathlib import Path
from research_extension.core import ROOT,load
from research_extension.contracts import apply_contract,sanitize_novelty
from research_extension.profiles import constrain_queries
class OutputSafety(unittest.TestCase):
    def test_uncited_fact_retracted_without_changing_raw_model(self):
        raw={"claims":[{"statement":"Already proven","type":"fact","evidence_ids":[]}],"mechanism":"Candidate mechanism","nearest_work_difference":"Unknown","risks":"Unknown"}
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            r=Path(tmp);out=apply_contract(raw,{"provenance":"test protocol"},r,[],[])
            self.assertEqual(out["claims"][0]["type"],"hypothesis")
            self.assertEqual(raw["claims"][0]["type"],"fact")
            self.assertEqual(load(r/"model_final_plan.json")["claims"][0]["type"],"fact")
    def test_foreign_paper_quote_cannot_support_nearest_work(self):
        novelty={"verdict":"high_overlap","scope_limitation":"","comparisons":[{"paper_id":"B","evidence_ids":["A-E1"],"existing_problem":"unsupported","existing_mechanism":"unsupported","existing_setting":"unsupported","candidate_difference":"unsupported"}]}
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            checked=sanitize_novelty(novelty,[{"evidence_id":"A-E1","paper_id":"A"}],Path(tmp))
            self.assertEqual(checked["comparisons"][0]["evidence_ids"],[])
            self.assertEqual(checked["verdict"],"insufficient_evidence")
            self.assertEqual(novelty["comparisons"][0]["evidence_ids"],["A-E1"])
    def test_internal_ids_removed_from_search_query(self):
        q=constrain_queries(["P6f6a2b14ed73 acoustic emission"],"电池热失控预警问题")[0]
        self.assertNotIn("P6f6a2b14ed73",q)
if __name__=="__main__":unittest.main()
