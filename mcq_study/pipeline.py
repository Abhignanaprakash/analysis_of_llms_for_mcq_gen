"""Dataset assembly and fixed split creation."""
from __future__ import annotations
import argparse, hashlib, html, random, re
from pathlib import Path
from bs4 import BeautifulSoup
from datasets import load_dataset
from rapidfuzz.fuzz import ratio
from .common import INSTRUCTION, jsonl, load_config, seed_everything, write_jsonl

LABELS = "ABCD"
def clean(text):
    text=BeautifulSoup(html.unescape(str(text)), "html.parser").get_text(" ")
    return re.sub(r"\s+", " ", text.encode("utf-8", "ignore").decode("utf-8")).strip()
def normalize(row):
    opts=[clean(x) for x in row["options"]][:4]
    if len(opts)!=4 or not all(opts) or len(set(x.casefold() for x in opts)) != 4: return None
    answer=str(row["answer"]).strip().upper()
    if answer in LABELS: answer=LABELS.index(answer)
    elif answer.isdigit(): answer=int(answer)
    else:
        matches=[i for i,x in enumerate(opts) if x.casefold()==answer.casefold()]
        if len(matches)!=1: return None
        answer=matches[0]
    if not isinstance(answer,int) or answer not in range(4): return None
    context=clean(row.get("context", "")); question=clean(row.get("question", ""))
    if not context or not question: return None
    identifier=row.get("id") or hashlib.sha256((context+question).encode()).hexdigest()[:16]
    # Deterministic option permutation avoids a source-specific answer-position bias.
    paired=list(enumerate(opts)); random.Random(identifier).shuffle(paired)
    answer=next(i for i,(old,_) in enumerate(paired) if old==answer)
    return {"id": identifier,
      "context":context, "question":question, "options":[x for _,x in paired], "answer":LABELS[answer],
      "subject":clean(row.get("subject", "unknown")).lower(), "source":row.get("source", "unknown"),
      "instruction":INSTRUCTION.format(context=context)}
def sciq_rows():
    ds=load_dataset("allenai/sciq", split="train")
    for x in ds:
        yield {"context":x["support"], "question":x["question"], "options":[x["correct_answer"],x["distractor1"],x["distractor2"],x["distractor3"]], "answer":0,"source":"SciQ"}
def arc_rows():
    for split in ("train", "validation", "test"):
        for x in load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split):
            choices=x["choices"]; yield {"context":x.get("question", ""),"question":x["question"],"options":choices["text"],"answer":x["answerKey"],"source":"ARC-Challenge"}
def openbook_rows():
    for split in ("train", "validation", "test"):
        for x in load_dataset("allenai/openbookqa", "main", split=split):
            choices=x["choices"]; yield {"context":x["question_stem"],"question":x["question_stem"],"options":choices["text"],"answer":x["answerKey"],"source":"OpenBookQA"}
def deduplicate(rows, threshold=96):
    kept=[]; buckets={}
    for row in rows:
        key=hashlib.sha1(re.sub(r"\W", "", row["context"].casefold()).encode()).hexdigest()[:8]
        candidates=buckets.setdefault(key, [])
        text=row["context"]+" "+row["question"]
        if any(ratio(text, old["context"]+" "+old["question"]) >= threshold for old in candidates): continue
        candidates.append(row); kept.append(row)
    return kept
def render_target(r):
    return "Question: {question}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nCorrect Answer: {answer}".format(question=r["question"],a=r["options"][0],b=r["options"][1],c=r["options"][2],d=r["options"][3],answer=r["answer"])
def main(args):
    cfg=load_config(args.config); seed_everything(cfg["seed"]); raw=[]
    for provider in (sciq_rows,arc_rows,openbook_rows): raw.extend(provider())
    custom=Path(cfg["paths"]["custom_data"])
    if custom.exists(): raw.extend(jsonl(custom))
    else: print(f"WARNING: custom dataset missing: {custom}; add it before the final study.")
    cleaned=[x for x in (normalize(r) for r in raw) if x]; rows=deduplicate(cleaned)
    for r in rows: r["target"]=render_target(r)
    from datasets import Dataset
    ds=Dataset.from_list(rows).shuffle(seed=cfg["seed"])
    first=ds.train_test_split(test_size=cfg["split"]["test"], seed=cfg["seed"])
    val_fraction=cfg["split"]["validation"]/(cfg["split"]["train"]+cfg["split"]["validation"])
    second=first["train"].train_test_split(test_size=val_fraction, seed=cfg["seed"])
    out=Path(cfg["paths"]["processed"]); out.mkdir(parents=True,exist_ok=True)
    for name,part in {"train":second["train"],"validation":second["test"],"test":first["test"]}.items(): write_jsonl(out/f"{name}.jsonl",part)
    print(f"Wrote immutable split: train={len(second['train'])}, validation={len(second['test'])}, test={len(first['test'])}")
