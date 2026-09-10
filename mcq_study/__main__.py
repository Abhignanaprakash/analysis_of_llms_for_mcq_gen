from . import pipeline, train, generate, evaluate, stats
import argparse

parser=argparse.ArgumentParser(description="MCQ study")
parser.add_argument("command", choices=["build","train","generate","evaluate","stats","environment"])
parser.add_argument("target", nargs="?", default="")
parser.add_argument("--config", default="configs/study.yaml")
args=parser.parse_args()
dispatch={"build":pipeline.main,"train":train.main,"generate":generate.main,"evaluate":evaluate.main,"stats":stats.main,"environment":train.environment}
dispatch[args.command](args)
