"""Matched LoRA/QLoRA training, using each model's native tokenizer."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from .common import capture_environment, jsonl, load_config, seed_everything

def environment(args):
    cfg=load_config(args.config); out=Path(cfg["paths"]["outputs"]); out.mkdir(parents=True,exist_ok=True)
    (out/"table_iv_environment.json").write_text(json.dumps(capture_environment(),indent=2),encoding="utf-8")
    print(out/"table_iv_environment.json")
def chat_text(tokenizer,row):
    msgs=[{"role":"user","content":row["instruction"]},{"role":"assistant","content":row["target"]}]
    if getattr(tokenizer,"chat_template",None): return tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    return f"User: {row['instruction']}\nAssistant: {row['target']}"
def targets_available(model,names):
    actual={n.rsplit('.',1)[-1] for n,_ in model.named_modules()}
    missing=set(names)-actual
    if missing: raise ValueError(f"Target modules unavailable: {sorted(missing)}. Document an approved per-model override in config.")
def run_model(name,spec,cfg):
    seed_everything(cfg["seed"]); tr=cfg["training"]; qlora=spec["peft"].lower()=="qlora"
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16) if qlora else None
    tokenizer=AutoTokenizer.from_pretrained(spec["base_model"],token=True)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(spec["base_model"],token=True,quantization_config=quant,device_map="auto" if qlora else None,torch_dtype="auto")
    model.config.use_cache=False; targets_available(model,tr["target_modules"])
    model=get_peft_model(model,LoraConfig(r=tr["lora_rank"],lora_alpha=tr["lora_alpha"],lora_dropout=tr["lora_dropout"],target_modules=tr["target_modules"],bias="none",task_type="CAUSAL_LM"))
    data=Dataset.from_list(list(jsonl(Path(cfg["paths"]["processed"])/"train.jsonl"))).map(lambda x:{"text":chat_text(tokenizer,x)})
    val=Dataset.from_list(list(jsonl(Path(cfg["paths"]["processed"])/"validation.jsonl"))).map(lambda x:{"text":chat_text(tokenizer,x)})
    out=Path(cfg["paths"]["outputs"])/"adapters"/name
    ta=TrainingArguments(output_dir=str(out),num_train_epochs=tr["epochs"],per_device_train_batch_size=tr["per_device_batch_size"],gradient_accumulation_steps=tr["gradient_accumulation_steps"],learning_rate=tr["learning_rate"],optim=tr["optimizer"],lr_scheduler_type=tr["scheduler"],logging_steps=10,save_strategy="epoch",eval_strategy="epoch",bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),report_to="none",seed=cfg["seed"])
    trainer=SFTTrainer(model=model,tokenizer=tokenizer,train_dataset=data,eval_dataset=val,dataset_text_field="text",max_seq_length=cfg["max_seq_length"],args=ta)
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    trainer.train(); trainer.save_model(str(out)); tokenizer.save_pretrained(out)
    peak_training_gpu_memory_gb=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.0
    (out/"study_metadata.json").write_text(json.dumps({"model":name,"base_model":spec["base_model"],"peft":spec["peft"],"requested_targets":tr["target_modules"],"training":tr,"peak_training_gpu_memory_gb":peak_training_gpu_memory_gb,"environment":capture_environment()},indent=2),encoding="utf-8")
def main(args):
    cfg=load_config(args.config); targets=cfg["models"] if args.target in ("","all") else {args.target:cfg["models"][args.target]}
    environment(args)
    for name,spec in targets.items(): run_model(name,spec,cfg)
