"""Resave a trained SALAAD checkpoint in Hugging Face format."""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from salaad.operators import opt_replace, opt_slr
from salaad.register import get_model
from salaad.uia import UIA
from salaad.utils import get_lowspa_layers, load_model, mkdir, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="Directory containing model.pth, config files, and matrix_rank*.pkl files.")
    parser.add_argument("--cfg_version", default=None,
                        help="Config stem. Defaults to the single YAML file in run_dir.")
    parser.add_argument("--output_dir", default=None,
                        help="Directory for Hugging Face model folders. Defaults to <run_dir>/model_resave.")
    parser.add_argument("--target_params", type=float, default=None,
                        help="Target parameter count in millions for the surrogate model.")
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Fraction of compression assigned to the low-rank component.")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--tokenizer", default="t5-base")
    parser.add_argument("--skip_surrogate", action="store_true",
                        help="Only save the vanilla checkpoint.")
    return parser.parse_args()


def infer_cfg_version(run_dir: str) -> str:
    cfg_candidates = [
        f[:-5] for f in os.listdir(run_dir)
        if f.endswith(".yaml")
    ]
    if len(cfg_candidates) != 1:
        raise ValueError(
            "Could not infer cfg_version from run_dir. Pass --cfg_version explicitly."
        )
    return cfg_candidates[0]


def parse_precision(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def load_low_sparse_components(run_dir: str, device):
    LL = {}
    SS = {}
    rank_files = sorted(f for f in os.listdir(run_dir) if f.startswith("matrix"))
    if not rank_files:
        raise FileNotFoundError(f"No matrix_rank*.pkl files found in {run_dir}")

    for filename in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(run_dir, filename))
        for key in LL_part:
            if "lm_head" in key:
                LL[key] = LL_part[key].to(device).t()
                SS[key] = SS_part[key].to(device).t()
            else:
                LL[key] = LL_part[key].to(device)
                SS[key] = SS_part[key].to(device)
    return LL, SS


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg_version = args.cfg_version or infer_cfg_version(run_dir)
    output_dir = args.output_dir or os.path.join(run_dir, "model_resave")
    precision = parse_precision(args.precision)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_cfg = os.path.join(run_dir, f"{cfg_version}.yaml")
    path_cfg_model = os.path.join(run_dir, f"{cfg_version}_model.json")

    with open(path_cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    max_length = cfg["max_length"]

    model = get_model(path_cfg_model)
    model.to(precision)
    load_model(model, os.path.join(run_dir, "model.pth"))
    model.to(device)

    mkdir(output_dir)
    vanilla_dir = os.path.join(output_dir, "vanilla")
    mkdir(vanilla_dir)
    model.save_pretrained(vanilla_dir, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=max_length)
    tokenizer.save_pretrained(vanilla_dir)
    print(f"Vanilla model saved to: {vanilla_dir}")

    if args.skip_surrogate:
        return

    if args.target_params is None:
        raise ValueError("--target_params is required unless --skip_surrogate is set.")

    LL, SS = load_low_sparse_components(run_dir, device)
    with open(os.path.join(run_dir, "layer_info.pkl"), "rb") as f:
        layer_info = pickle.load(f)

    uia = UIA(LL, SS, model, layer_info=layer_info, rate=100000000.0, rank=0)
    layers = [entry["name"] for entry in cfg["layers"]]
    gamma = float(np.clip(args.gamma, 0, 1))

    raw_rank_quantile, raw_rate_density, return_flag = uia.allocate(
        params_tgt=args.target_params,
        gamma=gamma,
    )
    rank_quantile, rate_density = uia.post_allocate(
        raw_rank_quantile,
        raw_rate_density,
        params_tgt=args.target_params,
    )

    nr_params = uia.check_params(rank_quantile, rate_density)
    print(f"Surrogate parameters: {nr_params / 1e6:.2f}M")
    print(f"Target parameters: {args.target_params:.2f}M")
    print(f"UIA return flag: {return_flag}")

    XX = opt_slr(LL, SS, rank_quantile, rate_density, layers, device)
    opt_replace(model, layers, XX, device)

    surrogate_dir = os.path.join(output_dir, "surrogate")
    mkdir(surrogate_dir)
    model.save_pretrained(surrogate_dir, safe_serialization=True)
    tokenizer.save_pretrained(surrogate_dir)
    print(f"Surrogate model saved to: {surrogate_dir}")


if __name__ == "__main__":
    main()
