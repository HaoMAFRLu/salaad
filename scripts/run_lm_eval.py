"""Run LM Evaluation Harness on a resaved Hugging Face model folder."""
import argparse
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from salaad.utils import hf_login_once, mkdir


DEFAULT_TASKS = [
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "boolq",
    "copa",
    "mmlu",
    "hellaswag",
    "gsm8k",
    "truthfulqa",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True,
                        help="Directory produced by scripts/resave_model.py, or a direct HF model folder.")
    parser.add_argument("--variant", choices=["vanilla", "surrogate", "both", "direct"], default="vanilla",
                        help="Which model variant to evaluate.")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Defaults to <model_dir>/lm_harness_eval_results.")
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS,
                        help="LM Evaluation Harness task names.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default=None,
                        help="Torch device string. Defaults to cuda if available, otherwise cpu.")
    parser.add_argument("--allow_code_eval", action="store_true",
                        help="Set HF_ALLOW_CODE_EVAL=1 for tasks that require code execution.")
    return parser.parse_args()


def resolve_dtype(name: str):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def variant_paths(model_dir: str, variant: str):
    if variant == "direct":
        return {"direct": model_dir}
    if variant == "both":
        return {
            "vanilla": os.path.join(model_dir, "vanilla"),
            "surrogate": os.path.join(model_dir, "surrogate"),
        }
    return {variant: os.path.join(model_dir, variant)}


def main():
    args = parse_args()
    if args.allow_code_eval:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    hf_login_once()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(args.dtype)
    output_dir = args.output_dir or os.path.join(args.model_dir, "lm_harness_eval_results")
    mkdir(output_dir)

    for label, model_path in variant_paths(os.path.abspath(args.model_dir), args.variant).items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = HFLM(
            pretrained=model_path,
            dtype=dtype,
            device=device,
            batch_size=args.batch_size,
        )
        results = evaluator.simple_evaluate(
            model=model,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
        )

        output_path = os.path.join(output_dir, f"results_{label}.pkl")
        with open(output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
