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
def bloom_llm(cfg, generated_question, reference_question):
    """Classify both questions with a versioned structured LLM judgement."""
    if cfg["evaluation"]["bloom_provider"] != "openai":
        raise ValueError("Bloom alignment requires evaluation.bloom_provider: openai; heuristic labels are not report-valid.")
    import os
    if not os.getenv("OPENAI_API_KEY"): raise EnvironmentError("Set OPENAI_API_KEY before LLM-assisted Bloom evaluation.")
    from openai import OpenAI
    schema={"type":"object","properties":{"generated_level":{"type":"integer","enum":[1,2,3,4]},"reference_level":{"type":"integer","enum":[1,2,3,4]},"generated_label":{"type":"string","enum":["recall","understand","apply","analyse"]},"reference_label":{"type":"string","enum":["recall","understand","apply","analyse"]}},"required":["generated_level","reference_level","generated_label","reference_label"],"additionalProperties":False}
    prompt=("Classify each MCQ question using this four-level Bloom rubric: 1 recall, 2 understand, 3 apply, 4 analyse. Classify the cognitive operation required to answer it, not its topic.\n\nGenerated question:\n"+generated_question+"\n\nReference question:\n"+reference_question)
    response=OpenAI().responses.create(model=cfg["evaluation"]["bloom_model"],input=prompt,store=False,text={"format":{"type":"json_schema","name":"bloom_pair","strict":True,"schema":schema}})
    return json.loads(response.output_text)
def entailment_pipeline(cfg):
    from transformers import pipeline
    import torch
    return pipeline("text-classification",model=cfg["evaluation"]["entailment_model"],top_k=None,device=0 if torch.cuda.is_available() else -1)
def hallucination_rate(cfg, classifier, context, generated):
    predictions=classifier({"text":context,"text_pair":generated[:1500]})
    entailment=next((p["score"] for p in predictions if "entail" in p["label"].lower()),0.0)
    return float(entailment < cfg["evaluation"]["entailment_threshold"])
def automated(cfg):
    try:
        import language_tool_python; tool=language_tool_python.LanguageTool(cfg["evaluation"]["language"])
    except Exception as exc: raise RuntimeError("LanguageTool requires Java; install it before automated evaluation.") from exc
    encoder=SentenceTransformer(cfg["evaluation"]["sentence_model"]); nli=entailment_pipeline(cfg)
    rows=[]; generation_dir=Path(cfg["paths"]["outputs"])/"generations"
    for file in generation_dir.glob("*.csv"):
        for r in csv.DictReader(open(file,encoding="utf-8")):
            question,opts,answer=parse(r["generated"]); options=list(opts.values())
            sim=cosine_similarity(encoder.encode(options)) if len(options)==4 else np.zeros((4,4))
            # Mean distance among distractors and between distractors and correct answer.
            distractor=1-float((sim.sum()-np.trace(sim))/(len(options)*(len(options)-1))) if len(options)==4 else 0.0
            bloom=bloom_llm(cfg,question,r["reference_question"])
            rows.append({**r,"parsed_answer":answer,"accuracy":float(answer==r["reference_answer"]),"grammar_quality":grammar_score(tool,r["generated"]),"bloom_level":bloom["generated_level"],"bloom_label":bloom["generated_label"],"reference_bloom_level":bloom["reference_level"],"reference_bloom_label":bloom["reference_label"],"bloom_alignment":float(bloom["generated_level"]==bloom["reference_level"]),"distractor_quality":1+4*distractor,"diversity":distinct_n(r["generated"]),"hallucination_rate":hallucination_rate(cfg,nli,r["context"],r["generated"])})
    frame=pd.DataFrame(rows); out=Path(cfg["paths"]["outputs"])/"metrics"; out.mkdir(parents=True,exist_ok=True); frame.to_csv(out/"automated_metrics.csv",index=False)
    print(out/"automated_metrics.csv")
def run_entailment(cfg):
    """Fill per-item unsupported-claim flags using the configured MNLI threshold."""
    p=entailment_pipeline(cfg)
    path=Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"; df=pd.read_csv(path)
    values=[]
    for _,r in df.iterrows():
        values.append(hallucination_rate(cfg,p,r["context"],r["generated"]))
    df["hallucination_rate"]=values; df.to_csv(path,index=False)
def human_form(cfg):
    df=pd.read_csv(Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"); dest=Path(cfg["paths"]["outputs"])/"human_rating_form.csv"
    df[["id","model","context","generated"]].assign(rater_id="", relevance_1to5="", clarity_1to5="", correctness_1to5="", distractor_quality_1to5="", comments="").to_csv(dest,index=False)
    print(f"Share {dest} with at least {cfg['evaluation']['human_raters_required']} independent raters. Keep model labels blinded if possible.")
def human_import(cfg):
    path=Path(cfg["paths"]["outputs"])/"human_rating_form.csv"; df=pd.read_csv(path); required=["rater_id","relevance_1to5","clarity_1to5","correctness_1to5"]
    required.append("distractor_quality_1to5")
    if df[required].isna().any().any(): raise ValueError("Rating form contains blank required fields.")
    rater_count=df.rater_id.nunique()
    if rater_count<cfg["evaluation"]["human_raters_required"] or rater_count>3: raise ValueError("Human evaluation requires 2-3 independent raters.")
    scores=["relevance_1to5","clarity_1to5","correctness_1to5","distractor_quality_1to5"]
    df[scores]=df[scores].apply(pd.to_numeric,errors="raise")
    df["human_score"]=df[scores[:3]].mean(axis=1); averages=df.groupby(["id","model"],as_index=False).agg(human_score=("human_score","mean"),human_distractor_quality=("distractor_quality_1to5","mean"))
    metrics=Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"; base=pd.read_csv(metrics).drop(columns=["human_score"],errors="ignore").merge(averages,on=["id","model"],how="left"); base.to_csv(metrics,index=False)
def main(args):
    cfg=load_config(args.config); action=args.target or "automated"
    {"automated":automated,"entailment":run_entailment,"human-form":human_form,"human-import":human_import}[action](cfg)
