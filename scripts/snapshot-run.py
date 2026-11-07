import hashlib,json,os,subprocess,sys
from pathlib import Path
r=Path(os.environ['WBT_ROOT']).resolve();run=Path(sys.argv[1]).resolve();run.relative_to(r)
(run/'requirements-lock.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
files={str(p.relative_to(r/'ResearchAgent-main')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'ResearchAgent-main/code').rglob('*.py')}
(run/'source-hashes.json').write_text(json.dumps(files,indent=2))
