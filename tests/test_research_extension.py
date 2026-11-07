import io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from research_extension.core import ROOT,inside,paper,deduplicate,rank,doi,safe_url,sha,dump
from research_extension.fulltext import verify_identity,parse_xml,parse_pdf,attachment_valid
from research_extension.retrieval import HTTP,Search
from research_extension.evidence import validate_references
from research_extension.llm import validate,obj,S,array
class Boundaries(unittest.TestCase):
    def test_project_boundary(self):
        with self.assertRaises(ValueError):inside(ROOT/"../outside")
    def test_symlink_boundary(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            p=Path(tmp)/"link";p.symlink_to(ROOT.parent,target_is_directory=True)
            with self.assertRaises(ValueError):inside(p/"anything")
    def test_private_network(self):
        for url in ("http://127.0.0.1/x","http://169.254.169.254/x","file:///etc/passwd","http://localhost/x"):
            with self.subTest(url=url),self.assertRaises(ValueError):safe_url(url)
class Identities(unittest.TestCase):
    def test_doi_normalization(self):self.assertEqual(doi("https://doi.org/10.1234/ABC"),"10.1234/abc")
    def test_merge_doi_sources(self):
        records=deduplicate([paper("a","Title",doi="10.1/a"),paper("b","Title",doi="10.1/A",abstract="Abstract")])
        self.assertEqual(len(records),1);self.assertEqual(records[0]["sources"],["a","b"]);self.assertEqual(records[0]["abstract"],"Abstract")
    def test_conflicting_dois_not_merged(self):
        self.assertEqual(len(deduplicate([paper("a","Same",doi="10.1/a"),paper("b","Same",doi="10.1/b")])),2)
    def test_arxiv_versions(self):
        self.assertEqual(len(deduplicate([paper("a","Paper v1",arxiv_id="2401.12345v1"),paper("b","Paper v2",arxiv_id="2401.12345v2")])),1)
    def test_ranking(self):
        docs=deduplicate([paper("a","Battery thermal runaway early warning"),paper("a","A study of galaxy clusters")])
        self.assertIn("Battery",rank(docs,"battery thermal runaway warning",1)[0]["title"])
    def test_document_identity(self):
        record=paper("a","Battery early warning",doi="10.1/a")
        self.assertFalse(verify_identity(record,"Battery early warning","10.1/b"))
class EvidenceAndFailures(unittest.TestCase):
    def test_xml_excludes_bibliography(self):
        raw=b'<article><front><article-meta><article-id pub-id-type="doi">10.1/a</article-id><abstract><p>ABSTRACT</p></abstract></article-meta></front><body><sec><title>Results</title><p>ACTUAL RESULT</p></sec></body><back><ref-list><ref><p>UNRELATED CITATION</p></ref></ref-list></back></article>'
        seg,details,links=parse_xml(raw,"https://example.org")
        self.assertEqual(details["document_doi"],"10.1/a");self.assertNotIn("UNRELATED"," ".join(x["text"] for x in seg))
    def test_html_not_pdf(self):
        with self.assertRaises(ValueError):parse_pdf(b"<html>login</html>")
    def test_html_not_si(self):
        with self.assertRaises(ValueError):attachment_valid(b"<html>login</html>","si.pdf","text/html")
    def test_unknown_ids_detected(self):
        issues=validate_references({"cited_paper_ids":["fake"],"claims":[{"statement":"Fact","type":"fact","evidence_ids":["fake-e"]}]},[],[])
        self.assertTrue(any(x["type"]=="unknown_paper_id" for x in issues));self.assertTrue(any(x["type"]=="unknown_evidence_id" for x in issues))
    def test_uncited_fact(self):
        issues=validate_references({"cited_paper_ids":[],"claims":[{"statement":"Fact","type":"fact","evidence_ids":[]}]},[],[])
        self.assertTrue(any(x["type"]=="uncited_fact" for x in issues))
    def test_schema_missing_fields(self):
        with self.assertRaises(ValueError):validate({},obj({"required":S}))
    def test_budget_exhaustion_no_network(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            h=HTTP(tmp,budget=0)
            with self.assertRaises(RuntimeError):h.get("https://example.org/a_unique_uncached_record")
    def test_source_failure_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            h=HTTP(tmp);s=Search(h)
            with patch.object(s,"query",side_effect=RuntimeError("HTTP 429")):
                records=s.run(["battery"],sources=["semantic_scholar"])
            self.assertEqual(records,[]);self.assertEqual(s.outcomes[0]["events"][0]["status"],"failed")
    def test_atomic_unicode(self):
        with tempfile.TemporaryDirectory(dir=ROOT/"tmp") as tmp:
            p=Path(tmp)/"unicode.json";dump(p,{"问题":"电池热失控"})
            self.assertEqual(json.loads(p.read_text())["问题"],"电池热失控")
if __name__=="__main__":unittest.main(verbosity=2)
