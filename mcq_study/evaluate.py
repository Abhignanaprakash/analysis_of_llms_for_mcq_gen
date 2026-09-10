"""Automated metrics plus export/import of blinded independent human ratings."""
from __future__ import annotations
import csv, json, re
from pathlib import Path
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .common import load_config

ANSWER=re.compile(r"(?:correct\s*answer|answer)\s*:\s*([ABCD])\b",re.I)
OPTION=re.compile(r"^[ ]*([ABCD])[.)]\s*(.+)$",re.M)
def parse(text):
    answer=ANSWER.search(text); opts=dict((a.upper(),b.strip()) for a,b in OPTION.findall(text))
    question=re.search(r"question\s*:\s*(.+?)(?=\n\s*A[.)])",text,re.I|re.S)
    return (question.group(1).strip() if question else text.split("\n")[0].strip(),opts,answer.group(1).upper() if answer else None)
def grammar_score(tool,text):
    words=max(1,len(re.findall(r"\b\w+\b",text))); errors=len(tool.check(text)); return float(np.clip(5-errors/words*25,1,5))
def distinct_n(text,n=2):
    words=re.findall(r"\w+",text.lower()); grams=list(zip(*[words[i:] for i in range(n)])); return len(set(grams))/len(grams) if grams else 0.0
def bloom_heuristic(question):
    q=question.lower(); verbs={"remember":["define","list","identify","what is","which"],"understand":["explain","summarize","describe"],"apply":["use","calculate","solve","implement"],"analyze":["compare","differentiate","analyze"],"evaluate":["justify","evaluate","assess"],"create":["design","create","develop"]}
    for i,(level,keys) in enumerate(verbs.items(),1):
        if any(k in q for k in keys): return i,level
    return 1,"remember"
def automated(cfg):
    try:
        import language_tool_python; tool=language_tool_python.LanguageTool(cfg["evaluation"]["language"])
    except Exception as exc: raise RuntimeError("LanguageTool requires Java; install it before automated evaluation.") from exc
    encoder=SentenceTransformer(cfg["evaluation"]["sentence_model"])
    rows=[]; generation_dir=Path(cfg["paths"]["outputs"])/"generations"
    for file in generation_dir.glob("*.csv"):
        for r in csv.DictReader(open(file,encoding="utf-8")):
            question,opts,answer=parse(r["generated"]); options=list(opts.values())
            sim=cosine_similarity(encoder.encode(options)) if len(options)==4 else np.zeros((4,4))
            # Mean distance among distractors and between distractors and correct answer.
            distractor=1-float((sim.sum()-np.trace(sim))/(len(options)*(len(options)-1))) if len(options)==4 else 0.0
            level,label=bloom_heuristic(question)
            rows.append({**r,"parsed_answer":answer,"accuracy":float(answer==r["reference_answer"]),"grammar_quality":grammar_score(tool,r["generated"]),"bloom_level":level,"bloom_label":label,"bloom_alignment":float(level>0),"distractor_quality":distractor,"diversity":distinct_n(r["generated"]),"hallucination_rate":np.nan})
    # Entailment / hallucination score: NLI is optional because it is expensive; record NA until explicitly run.
    # `run_entailment` below makes the method and model traceable rather than treating BERTScore as factual proof.
    frame=pd.DataFrame(rows); out=Path(cfg["paths"]["outputs"])/"metrics"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"automated_metrics.csv",index=False)
    print(out/"automated_metrics.csv")
def run_entailment(cfg):
    """Fill hallucination_rate using an MNLI model: 1 - P(context entails generated question)."""
    from transformers import pipeline
    p=pipeline("text-classification",model=cfg["evaluation"]["bertscore_model"],top_k=None,device=0 if __import__('torch').cuda.is_available() else -1)
    path=Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"; df=pd.read_csv(path)
    values=[]
    for _,r in df.iterrows():
        pred=p({"text":r["context"],"text_pair":r["generated"][:1000]})
        entail=next((x["score"] for x in pred if x["label"].upper().startswith("ENTAIL")),0.0); values.append(1-entail)
    df["hallucination_rate"]=values; df.to_csv(path,index=False)
def human_form(cfg):
    df=pd.read_csv(Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"); dest=Path(cfg["paths"]["outputs"])/"human_rating_form.csv"
    df[["id","model","context","generated"]].assign(rater_id="", relevance_1to5="", clarity_1to5="", correctness_1to5="", comments="").to_csv(dest,index=False)
    print(f"Share {dest} with at least {cfg['evaluation']['human_raters_required']} independent raters. Keep model labels blinded if possible.")
def human_import(cfg):
    path=Path(cfg["paths"]["outputs"])/"human_rating_form.csv"; df=pd.read_csv(path); required=["rater_id","relevance_1to5","clarity_1to5","correctness_1to5"]
    if df[required].isna().any().any(): raise ValueError("Rating form contains blank required fields.")
    if df.rater_id.nunique()<cfg["evaluation"]["human_raters_required"]: raise ValueError("Fewer than required independent raters.")
    df["human_score"]=df[["relevance_1to5","clarity_1to5","correctness_1to5"]].mean(axis=1); averages=df.groupby(["id","model"],as_index=False).human_score.mean()
    metrics=Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"; base=pd.read_csv(metrics).drop(columns=["human_score"],errors="ignore").merge(averages,on=["id","model"],how="left"); base.to_csv(metrics,index=False)
def main(args):
    cfg=load_config(args.config); action=args.target or "automated"
    {"automated":automated,"entailment":run_entailment,"human-form":human_form,"human-import":human_import}[action](cfg)
