"""Paired inferential tests and quality-to-compute frontier."""
from __future__ import annotations
from itertools import combinations
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from .common import load_config

def ci(a, rng, n=10000):
    a=np.asarray(a,float); means=[rng.choice(a,len(a),replace=True).mean() for _ in range(n)]
    return np.quantile(means,[.025,.975])
def cohens_d_paired(a,b):
    d=np.asarray(a)-np.asarray(b); return d.mean()/d.std(ddof=1) if len(d)>1 and d.std(ddof=1)>0 else np.nan
def main(args):
    cfg=load_config(args.config); source=Path(cfg["paths"]["outputs"])/"metrics"/"automated_metrics.csv"; df=pd.read_csv(source)
    out=source.parent/"statistics"; out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(cfg["seed"]); alpha=cfg["statistics"]["alpha"]
    metrics=[m for m in cfg["statistics"]["metric_directions"] if m in df and df[m].notna().any()]
    desc=[]; pairs=[]; anova=[]
    for metric in metrics:
        complete=df[["id","model",metric]].dropna().pivot(index="id",columns="model",values=metric).dropna()
        for model in complete:
            x=complete[model].to_numpy(); lo,hi=ci(x,rng,cfg["statistics"]["bootstrap_iterations"])
            desc.append({"metric":metric,"model":model,"n":len(x),"mean":x.mean(),"median":np.median(x),"variance":np.var(x,ddof=1),"std_dev":np.std(x,ddof=1),"ci95_low":lo,"ci95_high":hi})
        omnibus_significant=False
        if len(complete.columns)>2 and len(complete)>1:
            long=complete.reset_index().melt(id_vars="id",var_name="model",value_name="value")
            try:
                tab=AnovaRM(long,"value","id",within=["model"]).fit().anova_table.iloc[0]
                omnibus_significant=bool(tab["Pr > F"]<alpha)
                anova.append({"metric":metric,"F":tab["F Value"],"df_num":tab["Num DF"],"df_den":tab["Den DF"],"p":tab["Pr > F"],"significant":omnibus_significant})
            except ValueError: pass
        # Pre-registered post-hoc tests are run only after a significant
        # repeated-measures one-factor ANOVA, controlling familywise error.
        if omnibus_significant:
            tests=[]
            for a,b in combinations(complete.columns,2):
                t,p=stats.ttest_rel(complete[a],complete[b]); tests.append([a,b,t,p,cohens_d_paired(complete[a],complete[b])])
            corrected=multipletests([x[3] for x in tests],alpha=alpha,method="bonferroni")[1]
            for x,padj in zip(tests,corrected): pairs.append({"metric":metric,"model_a":x[0],"model_b":x[1],"t":x[2],"p_raw":x[3],"p_bonferroni":padj,"significant":padj<alpha,"cohens_d":x[4]})
    desc=pd.DataFrame(desc); desc.to_csv(out/"descriptive_and_ci.csv",index=False); pd.DataFrame(pairs).to_csv(out/"paired_ttests_bonferroni_cohens_d.csv",index=False); pd.DataFrame(anova).to_csv(out/"repeated_measures_anova.csv",index=False)
    # Metric correlations use all matched observations, retaining Pearson and rank-based Spearman views.
    numerical=df[metrics].apply(pd.to_numeric,errors="coerce"); pearson=numerical.corr("pearson"); spearman=numerical.corr("spearman"); pearson.to_csv(out/"metric_correlations_pearson.csv"); spearman.to_csv(out/"metric_correlations_spearman.csv")
    # The report permits a composite ranking only after inferential analysis.
    # Include only metrics with at least one significant, non-undefined paired
    # effect size; otherwise publish the reason rather than an arbitrary rank.
    pair_df=pd.DataFrame(pairs)
    eligible_metrics=set(pair_df.loc[pair_df.significant & pair_df.cohens_d.notna(),"metric"]) if not pair_df.empty else set()
    rankings=[]
    for metric, direction in cfg["statistics"]["metric_directions"].items():
        if metric not in eligible_metrics: continue
        sub=desc[desc.metric==metric]
        if sub.empty: continue
        ranks=sub.set_index("model")["mean"].rank(ascending=direction=="lower",method="average")
        for model,rank in ranks.items(): rankings.append({"model":model,"metric":metric,"rank":rank,"weight":cfg["statistics"]["deployment_weights"].get(metric,1)})
    rank=pd.DataFrame(rankings)
    if not rank.empty:
        composite=rank.assign(weighted=lambda x:x["rank"]*x["weight"]).groupby("model").agg(weighted_rank=("weighted","sum"),weight_total=("weight","sum")).reset_index(); composite["composite_rank_score"]=composite.weighted_rank/composite.weight_total; composite.sort_values("composite_rank_score").to_csv(out/"composite_ranking.csv",index=False)
    else:
        (out/"composite_ranking_not_performed.txt").write_text("No metric had a statistically significant paired comparison with a defined Cohen's d. A composite rank would violate the pre-registered analysis rule.\n",encoding="utf-8")
    # Pareto frontier: do not claim a universal winner; maximize quality, minimize response time and memory.
    quality=desc[desc.metric.isin(["accuracy","grammar_quality","bloom_alignment","distractor_quality","human_score"])].groupby("model").mean(numeric_only=True)["mean"].rename("mean_quality")
    compute=desc[desc.metric.isin(["response_time_s","peak_gpu_memory_mb"])].pivot(index="model",columns="metric",values="mean")
    frontier=quality.to_frame().join(compute).reset_index()
    # A model is Pareto efficient when no other model is at least as good on all
    # three deployment axes and strictly better on one.
    frontier["pareto_efficient"]=True
    for i,row in frontier.iterrows():
        others=frontier.drop(index=i)
        dominated=((others.mean_quality>=row.mean_quality)&(others.response_time_s<=row.response_time_s)&(others.peak_gpu_memory_mb<=row.peak_gpu_memory_mb)&((others.mean_quality>row.mean_quality)|(others.response_time_s<row.response_time_s)|(others.peak_gpu_memory_mb<row.peak_gpu_memory_mb))).any()
        frontier.loc[i,"pareto_efficient"]=not dominated
    frontier.to_csv(out/"quality_compute_tradeoff.csv",index=False)
    print(f"Statistics written to {out}; interpret composite ranks only alongside significant paired tests and effect sizes.")
