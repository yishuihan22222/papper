"""Validate saved evidence, source preservation, local links, and delivery hashes."""
from pathlib import Path
import ast
import csv
import hashlib
import json
import re

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parents[1]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    first=json.loads((OUT/"summary.json").read_text(encoding="utf-8"))
    second=json.loads((OUT/"uncertain_capacity_summary.json").read_text(encoding="utf-8"))
    formula_checks=0
    with (OUT/"uncertain_capacity_grid.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            caps=json.loads(row["calendars"]);rs=json.loads(row["recoveries"])
            q=max(0,int(row["batch"])-max(0,4-max(rs)))
            available=min(sum(c[int(row["lag"]):]) for c in caps)
            assert (available>=q)==(row["feasible"]=="True")
            formula_checks+=1
    for p in (OUT/"code").glob("*.py"):
        ast.parse(p.read_text(encoding="utf-8"))
    for p in OUT.glob("*.json"):
        json.loads(p.read_text(encoding="utf-8-sig"))
    inputs=json.loads((ROOT/"outputs/researchwrite/new_topic_review_20261022/qa_logs/network_support_audit.json").read_text(encoding="utf-8"))
    preserved={p:sha(Path(p))==h for p,h in inputs["source_sha256"].items()}
    assert all(preserved.values())
    reports=[ROOT/"主线第一轮试跑_20261022.md",ROOT/"主线第二轮试跑_未来接收余量不确定_20261022.md"]
    links=[]
    for p in [*OUT.glob("*.md"),*reports]:
        for target in re.findall(r"\]\((E:/[^)]+)\)",p.read_text(encoding="utf-8")):
            assert Path(target).exists(),target
            links.append(target)
    paths=[p for p in OUT.rglob("*") if p.is_file() and "__pycache__" not in p.parts
           and p.name!="delivery_manifest.json"]+reports
    manifest=dict(completed_scope="Two bounded feasibility pilots; innovation unresolved",
                  phase1_configurations=first["configurations"],
                  phase2_configurations=second["configurations"],
                  phase2_closed_form_feasibility_checks=formula_checks,
                  original_data_preserved=preserved,
                  local_links_checked=len(links),
                  files={str(p):sha(p) for p in sorted(paths)})
    (OUT/"delivery_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in ("files","original_data_preserved")}))


if __name__=="__main__":
    main()
