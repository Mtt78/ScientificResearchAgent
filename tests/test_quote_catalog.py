import unittest
from research_extension.evidence import quote_catalog
class Quotes(unittest.TestCase):
    def test_catalog_is_always_literal(self):
        text="We detect battery thermal runaway before rapid heating with a causal predictor. The experiments use independent cell-level test splits."
        doc={"segments":[{"segment_id":"P-S1","locator":"Methods","text":text}]}
        catalog=quote_catalog(doc,"battery thermal runaway warning")
        self.assertTrue(catalog)
        for row in catalog:self.assertIn(row["quote"],text)
    def test_bibliography_not_used(self):
        text="An experiment was conducted with separate cells. REFERENCES [1] Battery thermal runaway warning gas detection lead time false alarm."
        doc={"segments":[{"segment_id":"P-S1","locator":"PDF file page 9","text":text}]}
        rows=quote_catalog(doc,"battery thermal runaway warning")
        self.assertFalse(any("REFERENCES" in x["quote"] for x in rows))
if __name__=="__main__":unittest.main()
