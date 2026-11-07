import io,tempfile,unittest,zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from research_extension.core import ROOT,paper,deduplicate
from research_extension.profiles import constrain_queries,topic_filter,ground_task
from research_extension.fulltext import supplementary_links,parse_html,parse_xml
from research_extension.llm import validate
class Regressions(unittest.TestCase):
    def test_query_preserves_both_concepts(self):
        q=constrain_queries(["lead time failure prediction"],"电池热失控预警问题",initial=True)
        self.assertIn("battery thermal runaway",q[0]);self.assertIn("battery",q[1]);self.assertIn("thermal runaway",q[1])
    def test_exclude_aiops_and_binder_drying(self):
        docs=deduplicate([paper("x","Lead-time failure prediction for AIOps"),paper("x","Binder drying in lithium-ion battery electrodes"),paper("x","Battery thermal runaway warning")])
        kept,excluded=topic_filter(docs,"电池热失控预警问题")
        self.assertEqual(len(kept),1);self.assertEqual(len(excluded),2)
    def test_no_invented_user_facts(self):
        card=ground_task({"observed_facts":"实验已发现温升","constraints":"已具备气体传感器","unknowns":"未提供"},"电池热失控预警问题","problem")
        self.assertNotIn("已具备",card["constraints"]);self.assertIn("未提供实验观测",card["observed_facts"])
    def test_stable_id_after_doi_enrichment(self):
        old=deduplicate([paper("a","Exact title")])[0];pid=old["paper_id"]
        newer=deduplicate([old,paper("b","Exact title",doi="10.1/new")])[0]
        self.assertEqual(newer["paper_id"],pid)
    def test_conflicting_authors_not_merged_by_title(self):
        docs=[paper("a","Same title",authors=["Alice Smith"]),paper("b","Same title",authors=["Bob Wang"])]
        self.assertEqual(len(deduplicate(docs)),2)
    def test_si_filename_without_supplement_word(self):
        soup=BeautifulSoup('<section id="paper-supinfo-1"><a href="/bin/file-s001.docx">Click here for additional data</a></section>',"html.parser")
        links=supplementary_links(soup,"https://pmc.ncbi.nlm.nih.gov/articles/PMC123")
        self.assertEqual(len(links),1);self.assertTrue(links[0]["url"].endswith(".docx"))
    def test_abstract_only_html_rejected(self):
        raw=b"<html><article><h1>Title</h1><h2>Abstract</h2><p>"+b"A"*1000+b"</p></article></html>"
        with self.assertRaises(ValueError):parse_html(raw,"https://example.org")
    def test_abstract_only_xml_rejected(self):
        with self.assertRaises(ValueError):parse_xml(b"<article><front><abstract><p>Abstract</p></abstract></front></article>","https://example.org")
    def test_enum_checked_on_cache_replay(self):
        with self.assertRaises(ValueError):validate("invented",{"type":"string","enum":["fact","hypothesis"]})
if __name__=="__main__":unittest.main(verbosity=2)
