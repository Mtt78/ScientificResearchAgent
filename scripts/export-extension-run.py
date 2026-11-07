"""Export CSV/JSONL tables and independently check saved run artifacts."""
import csv,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_extension.core import ROOT,inside,load,dump,sha
from research_extension.evidence import validate_references
run=inside(sys.argv[1]);status=load(run/"status.json")
if status["state"] not in ("complete","complete_with_gaps"):raise RuntimeError("Run has not completed")
final=load(run/"final_plan.json");papers=load(run/"retrieved_papers.json");cards=load(run/"evidence_cards.json")
issues=validate_references(final,papers,cards)
known={x["paper_id"]:x for x in papers}
for c in cards:
    segment_path=run/"literature"/c["paper_id"]/"segments.json"
    segments=load(segment_path);seg=next((s for s in segments if s["segment_id"]==c["segment_id"]),None)
    if not seg or c["quote"] not in seg["text"]:issues.append({"type":"export_quote_location_failure","id":c["evidence_id"]})
    docs=list(segment_path.parent.glob("main.*"))
    if len(docs)!=1 or sha(docs[0].read_bytes())!=c["document_sha256"]:issues.append({"type":"export_document_hash_failure","id":c["evidence_id"]})
for m in load(run/"all_documents.json"):
    for a in m.get("supplements",[])+m.get("auxiliary_assets",[]):
        if a["status"]=="downloaded" and sha(inside(a["path"]).read_bytes())!=a["sha256"]:issues.append({"type":"attachment_hash_mismatch","path":a["path"]})
novelty=load(run/"novelty_audit.json");rows=novelty["comparisons"]
with (run/"novelty_matrix.csv").open("w",encoding="utf-8-sig",newline="") as f:
    fields=list(rows[0]) if rows else ["paper_id"]
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
with (run/"reviews.jsonl").open("w",encoding="utf-8") as f:
    for stage,review in load(run/"reviews.json").items():
        for item in review["reviews"]:f.write(json.dumps({"stage":stage,"reviewer_type":"same_local_model",**item},ensure_ascii=False)+"\n")
snapshot=load(run/"source-hashes.json")
mismatch=[name for name,digest in snapshot.items() if sha(inside(ROOT/name).read_bytes())!=digest]
if mismatch:issues.append({"type":"source_changed_since_snapshot","files":mismatch})
summary={"artifact_checks":"passed" if not issues else "needs_attention","issues":issues,"located_quotes":len(cards),"review_rows":sum(len(r["reviews"]) for r in load(run/"reviews.json").values()),"source_snapshot_matches_current":not mismatch,"scientific_verification":"requires_domain_review","timing_note":"Check status and logs for resume history; cached retrieval is not cold-start performance."}
dump(run/"artifact_validation.json",summary);print(json.dumps(summary,ensure_ascii=False,indent=2))
