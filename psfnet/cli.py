import argparse, yaml
from .train import run_train
from .eval import run_eval

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train-kernel")
    tr.add_argument("--config", type=str, required=True)

    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", type=str, required=True)
    ev.add_argument("--config", type=str, default=None)

    args = ap.parse_args()
    cfg = {}
    if getattr(args, "config", None):
        with open(args.config) as f: cfg = yaml.safe_load(f)

    if args.cmd == "train-kernel":
        run_train(cfg)
    else:
        run_eval(args.ckpt, cfg)

if __name__ == "__main__":
    main()
