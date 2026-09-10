"""Generate one MCQ per held-out source passage for each trained adapter."""
from __future__ import annotations
import csv, time
from pathlib import Path
import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from .common import jsonl, load_config

def prompt(tokenizer, instruction):
    msg=[{"role":"user","content":instruction}]
    return tokenizer.apply_chat_template(msg,tokenize=False,add_generation_prompt=True) if getattr(tokenizer,"chat_template",None) else f"User: {instruction}\nAssistant:"
def one_model(name,cfg):
    device="cuda" if torch.cuda.is_available() else "cpu"; directory=Path(cfg["paths"]["outputs"])/"adapters"/name
    model=AutoPeftModelForCausalLM.from_pretrained(directory,device_map="auto" if device=="cuda" else None,torch_dtype="auto").eval()
    tok=AutoTokenizer.from_pretrained(directory); tok.pad_token=tok.pad_token or tok.eos_token
    dest=Path(cfg["paths"]["outputs"])/"generations"; dest.mkdir(parents=True,exist_ok=True)
    with open(dest/f"{name}.csv","w",newline="",encoding="utf-8") as out:
        fields=["id","model","context","reference_question","reference_options","reference_answer","generated","response_time_s","peak_gpu_memory_gb","generated_tokens","tokens_per_second"]
        writer=csv.DictWriter(out,fieldnames=fields); writer.writeheader()
        for row in jsonl(Path(cfg["paths"]["processed"])/"test.jsonl"):
            if device=="cuda": torch.cuda.reset_peak_memory_stats()
            encoded=tok(prompt(tok,row["instruction"]),return_tensors="pt",truncation=True,max_length=cfg["max_seq_length"]).to(model.device)
            started=time.perf_counter(); output=model.generate(**encoded,max_new_tokens=cfg["generation"]["max_new_tokens"],do_sample=cfg["generation"]["do_sample"],temperature=cfg["generation"]["temperature"],pad_token_id=tok.pad_token_id,eos_token_id=tok.eos_token_id); elapsed=time.perf_counter()-started
            tokens=output.shape[1]-encoded["input_ids"].shape[1]
            writer.writerow({"id":row["id"],"model":name,"context":row["context"],"reference_question":row["question"],"reference_options":"|||".join(row["options"]),"reference_answer":row["answer"],"generated":tok.decode(output[0][encoded["input_ids"].shape[1]:],skip_special_tokens=True),"response_time_s":elapsed,"peak_gpu_memory_gb":torch.cuda.max_memory_allocated()/2**30 if device=="cuda" else 0,"generated_tokens":tokens,"tokens_per_second":tokens/elapsed if elapsed else 0})
def main(args):
    cfg=load_config(args.config); names=cfg["models"] if args.target in ("","all") else [args.target]
    for name in names: one_model(name,cfg)
