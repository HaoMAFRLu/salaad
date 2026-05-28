"""Evaluate a trained SALAAD run on C4 train/validation streams."""
import argparse
import copy
import os
import pickle
import sys

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from salaad.cross_evaluator import CrossEvaluator
from salaad.register import get_model
from salaad.utils import get_eval_data, get_lowspa_layers, load_model, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                        help="Directory containing model.pth, config files, and matrix_rank*.pkl files.")
    parser.add_argument("--cfg_version", default=None,
                        help="Config stem. Defaults to the basename of the run's parent directory.")
    parser.add_argument("--target_params", type=float, nargs="*", default=None,
                        help="Optional target parameter counts in millions for partial L/S evaluation.")
    parser.add_argument("--output", default=None,
                        help="Output pickle path. Defaults to <run_dir>/eval_results.pkl.")
    parser.add_argument("--tokenizer", default="t5-base",
                        help="Tokenizer name or path.")
    parser.add_argument("--eval_batch_size", type=int, default=10,
                        help="Batch size used by CrossEvaluator.")
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


def load_low_sparse_components(run_dir: str):
    LL = {}
    SS = {}
    rank_files = sorted(f for f in os.listdir(run_dir) if f.startswith("matrix"))
    if not rank_files:
        raise FileNotFoundError(f"No matrix_rank*.pkl files found in {run_dir}")

    for filename in rank_files:
        LL_part, SS_part = get_lowspa_layers(os.path.join(run_dir, filename))
        LL.update(LL_part)
        SS.update(SS_part)
    return LL, SS


def build_rank_schedules(model, cfg, LL, SS, target_params):
    layers = [entry["name"] for entry in cfg["layers"]]
    rank_quantile_target = {
        entry["name"]: entry["params"]["rate_rank"]
        for entry in cfg["layers"]
    }
    energy_quantile = {
        entry["name"]: entry["params"]["energy"]
        for entry in cfg["layers"]
    }

    rank_quantile_energy = {}
    rate_density = {}
    layer_dim = {}

    nr_params_model = sum(p.numel() for p in model.parameters())
    nr_params_layers = 0
    nr_params_L = 0
    nr_params_S = 0

    for name in energy_quantile:
        L = LL[name]
        S = SS[name]
        row, col = L.shape
        layer_dim[name] = (row, col)

        density = torch.count_nonzero(S).item() / S.numel()
        rate_density[name] = density

        _, singular_values, _ = torch.linalg.svd(L, full_matrices=False)
        energy = torch.cumsum(singular_values, dim=0) / torch.sum(singular_values)
        rank = torch.sum(energy < energy_quantile[name]).item() + 1
        rank_quantile_energy[name] = rank / len(singular_values)

        nr_params_layers += row * col
        nr_params_L += int(rank * (row + col))
        nr_params_S += int(torch.count_nonzero(S).item())

    nr_params_rest = nr_params_model - nr_params_layers
    nr_params_total = nr_params_rest + nr_params_L + nr_params_S

    if target_params is None:
        rank_quantile_list = [copy.deepcopy(rank_quantile_energy)]
    else:
        rank_quantile_list = []
        for target in target_params:
            params_to_remove = nr_params_total - target * 1e6
            if params_to_remove <= 0:
                rank_quantile_list.append(copy.deepcopy(rank_quantile_energy))
            elif params_to_remove >= nr_params_L:
                rank_quantile_list.append({name: 0.0 for name in rank_quantile_energy})
            else:
                ratio = 1 - params_to_remove / nr_params_L
                rank_quantile_list.append({
                    name: ratio * value
                    for name, value in rank_quantile_energy.items()
                })

    rate_density_list = [copy.deepcopy(rate_density) for _ in rank_quantile_list]
    return layers, rank_quantile_target, rank_quantile_energy, rank_quantile_list, rate_density, rate_density_list, layer_dim


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg_version = args.cfg_version or infer_cfg_version(run_dir)

    path_cfg = os.path.join(run_dir, f"{cfg_version}.yaml")
    path_cfg_model = os.path.join(run_dir, f"{cfg_version}_model.json")
    output = args.output or os.path.join(run_dir, "eval_results.pkl")

    with open(path_cfg, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    max_length = cfg["max_length"]
    batch_size = cfg["batch_size"]

    model = get_model(path_cfg_model)
    load_model(model, os.path.join(run_dir, "model.pth"))
    LL, SS = load_low_sparse_components(run_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=max_length)
    pad_idx = tokenizer.pad_token_id
    val_loader = get_eval_data(
        "validation",
        seed_for_shuffle=cfg["seed_for_shuffle"],
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )
    train_loader = get_eval_data(
        "train",
        seed_for_shuffle=cfg["seed_for_shuffle"],
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
    )

    (
        layers,
        rank_quantile_target,
        rank_quantile_energy,
        rank_quantile_list,
        rate_density,
        rate_density_list,
        layer_dim,
    ) = build_rank_schedules(model, cfg, LL, SS, args.target_params)

    evaluator = CrossEvaluator(
        model_type=cfg_version,
        model=model,
        train_loader=train_loader,
        test_loader=val_loader,
        layers=layers,
        pad_idx=pad_idx,
        LL=LL,
        SS=SS,
        layer_dim=layer_dim,
        batch_size=args.eval_batch_size,
    )
    evaluator.collect_model_results()
    evaluator.collect_single_results(rank_quantile_target, rate_density)
    evaluator.collect_results(rank_quantile_list, rate_density_list)

    data = {
        "eval_train_results": evaluator.eval_train_results,
        "eval_test_results": evaluator.eval_test_results,
        "rank_quantile_energy": rank_quantile_energy,
        "rate_density": rate_density,
        "target_params": args.target_params,
    }
    with open(output, "wb") as f:
        pickle.dump(data, f)
    print(f"Evaluation results saved to: {output}")


if __name__ == "__main__":
    main()
