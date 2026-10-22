"""Frozen paired GPU experiments and reproducible audits of historical records.

No weight or gate threshold is selected using the evaluation labels. New runs
and historical reconstructed metrics are always stored in separate tables.
"""

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("instance_auroc", "full_pixel_auroc", "anomaly_pixel_auroc")
ALIASES = {
    "fixed": "fixed_score",
    "image_fixed_pixel_sensitivity": "sensitivity_pixel",
    "image_fixed_pixel_sensitivity_shrinkage": "shrinkage_025",
    "gate_selected": "gate_legacy",
}


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_statistics(rows, baseline, seed=20261022):
    """Equal category weighting; bootstrap category means, NOT 75 iid seeds."""
    reference = {(r["category"], r["seed"]): r for r in rows if r["method"] == baseline}
    deltas, summaries, statistics = [], [], []
    methods = list(dict.fromkeys(r["method"] for r in rows))
    categories = sorted(set(r["category"] for r in rows))
    for method in methods:
        for category in categories:
            subset = [
                r for r in rows if r["method"] == method and r["category"] == category
            ]
            if not subset:
                continue
            for metric in METRICS:
                values = np.array([float(r[metric]) for r in subset])
                summaries.append(
                    dict(
                        category=category,
                        method=method,
                        metric=metric,
                        n=len(values),
                        mean=float(values.mean()),
                        std=float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        min=float(values.min()),
                        max=float(values.max()),
                    )
                )
        if method == baseline:
            continue
        for r in rows:
            if r["method"] != method:
                continue
            base = reference[(r["category"], r["seed"])]
            deltas.append(
                dict(
                    category=r["category"],
                    seed=r["seed"],
                    method=method,
                    baseline=baseline,
                    **{m: float(r[m]) - float(base[m]) for m in METRICS},
                )
            )
        for metric in METRICS:
            subset = [r for r in deltas if r["method"] == method]
            paired = np.array([r[metric] for r in subset])
            means = np.array(
                [
                    np.mean([r[metric] for r in subset if r["category"] == c])
                    for c in categories
                ]
            )
            rng = np.random.default_rng(seed)
            sampled = rng.choice(means, size=(20000, len(means)), replace=True).mean(
                axis=1
            )
            lo, hi = np.quantile(sampled, [0.025, 0.975])
            # Exact two-sided sign-flip test on the category-level paired means.
            signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(means))))
            distribution = (signs * means).mean(axis=1)
            p = float(np.mean(np.abs(distribution) >= abs(means.mean()) - 1e-15))
            statistics.append(
                dict(
                    method=method,
                    baseline=baseline,
                    metric=metric,
                    categories=len(means),
                    pairs=len(paired),
                    mean_delta=float(means.mean()),
                    pair_std=float(paired.std(ddof=1)),
                    wins=int(np.sum(paired > 1e-12)),
                    ties=int(np.sum(abs(paired) <= 1e-12)),
                    losses=int(np.sum(paired < -1e-12)),
                    category_bootstrap_95_low=float(lo),
                    category_bootstrap_95_high=float(hi),
                    category_signflip_p=p,
                )
            )
    return summaries, deltas, statistics


def audit_history(destination):
    source = ROOT / "results/experiment_2_20261022"
    prefix = "task_decoupled_shrinkage_full_20261022_"
    records = read_csv(source / (prefix + "gate_decisions.csv"))
    if len(records) != 75 or len({(r["category"], r["seed"]) for r in records}) != 75:
        raise ValueError("Expected 75 unique historical category/seed pairs.")
    sensitivity = {
        (r["category"], r["seed"]): r
        for r in read_csv(source / (prefix + "paired_deltas.csv"))
        if r["method"] == "image_fixed_pixel_sensitivity"
    }
    if not sensitivity:
        # Older table names the same pixel-only variant 'sensitivity'.
        candidates = read_csv(source / (prefix + "paired_deltas.csv"))
        names = sorted(set(r["method"] for r in candidates))
        print("Historical delta method names:", names, flush=True)
        sensitivity = {
            (r["category"], r["seed"]): r
            for r in candidates
            if "shrinkage" not in r["method"]
        }
    output, decisions = [], []
    for r in records:
        expected = (
            float(r["gate_group_consistency"]) >= 0.8
            and float(r["gate_layer2_weight"]) <= 0.8
            and float(r["gate_layer3_stability"]) >= 0.9
        )
        actual = r["gate_use_shrinkage"].lower() in ("true", "1")
        if expected != actual:
            raise ValueError(
                "Historical gate decision disagrees with frozen thresholds."
            )
        for metric in METRICS:
            chosen = r[("shrinkage_" if expected else "fixed_") + metric]
            if abs(float(chosen) - float(r["gate_" + metric])) > 1e-12:
                raise ValueError(
                    "Historical gate metric is not its selected branch metric."
                )
        for method in ("fixed", "shrinkage", "gate"):
            output.append(
                dict(
                    category=r["category"],
                    seed=int(r["seed"]),
                    method=method,
                    **{m: float(r[method + "_" + m]) for m in METRICS},
                )
            )
        delta = sensitivity[(r["category"], r["seed"])]
        output.append(
            dict(
                category=r["category"],
                seed=int(r["seed"]),
                method="sensitivity_pixel",
                **{
                    m: float(r["fixed_" + m]) + float(delta["delta_" + m])
                    for m in METRICS
                },
            )
        )
        decisions.append(
            dict(
                category=r["category"],
                seed=int(r["seed"]),
                gate_enabled=expected,
                decision_verified=True,
                selected_metrics_verified=True,
            )
        )
    write_csv(destination / "historical_reconstructed.csv", output)
    write_csv(destination / "historical_gate_audit.csv", decisions)
    summaries, deltas, stats = paired_statistics(output, "fixed")
    for name, rows in (
        ("summary", summaries),
        ("paired", deltas),
        ("statistics", stats),
    ):
        write_csv(destination / ("historical_" + name + ".csv"), rows)
    return stats


def command(category, seed, mode, args):
    group = f"{mode}_{category}_s{seed}"
    cmd = [
        sys.executable,
        str(ROOT / "bin/run_patchcore.py"),
        "--gpu",
        "0",
        "--seed",
        str(seed),
        "--log_project",
        args.output.name,
        "--log_group",
        group,
    ]
    if mode == "dual":
        cmd += [
            "--task_decoupled_ablation",
            "--pixel_sensitivity_shrinkage",
            "0.25",
            "--sensitivity_gate",
            "--gate_mode",
            "legacy",
            "--gate_min_group_consistency",
            "0.8",
            "--gate_max_layer2_weight",
            "0.8",
            "--gate_min_layer3_stability",
            "0.9",
            "--adaptive_max_samples",
            "64",
        ]
    cmd += [str(args.output.parent), "patch_core", "-b", "wideresnet50"]
    if mode == "dual":
        cmd += [
            "-b",
            "wideresnet50",
            "-le",
            "0.layer2",
            "-le",
            "1.layer3",
            "--image_fusion_weight",
            "0.9",
            "--image_fusion_weight",
            "0.1",
            "--pixel_fusion_weight",
            "0.5",
            "--pixel_fusion_weight",
            "0.5",
        ]
    else:
        cmd += ["-le", "layer2", "-le", "layer3"]
    cmd += [
        "--pretrain_embed_dimension",
        "1024",
        "--target_embed_dimension",
        "1024",
        "--anomaly_scorer_num_nn",
        "1",
        "--patchsize",
        "3",
        "sampler",
        "-p",
        "0.01",
        "approx_greedy_coreset",
        "dataset",
        "--train_val_split",
        "0.8",
        "--num_workers",
        "0",
        "--batch_size",
        "2",
        "--resize",
        "256",
        "--imagesize",
        "224",
        "-d",
        category,
        "mvtec",
        str(args.data),
    ]
    return group, cmd


def summarize_new(args):
    rows, timings, gates = [], [], []
    for category, seed in itertools.product(args.categories, args.seeds):
        manifests = []
        for mode in ("feature", "dual"):
            group, _ = command(category, seed, mode, args)
            folder = args.output / group
            config = json.loads((folder / f"mvtec_{category}_fusion.json").read_text())
            manifests.append(config["data_split"])
            for r in read_csv(folder / "results.csv"):
                if r["Row Names"].startswith("Mean"):
                    continue
                name = (
                    "feature_original"
                    if mode == "feature"
                    else ALIASES[r["Row Names"].split("__", 1)[1]]
                )
                rows.append(
                    dict(
                        category=category,
                        seed=seed,
                        method=name,
                        **{m: float(r[m]) for m in METRICS},
                    )
                )
            timings.append(
                dict(
                    category=category,
                    seed=seed,
                    mode=mode,
                    fit_s=config["fit_time_s"],
                    calibration_s=config["calibration_time_s"],
                    predict_s=config["predict_time_s"],
                    fit_calibration_predict_s=config["fit_calibration_predict_time_s"],
                    evaluation_s=config["evaluation_time_s"],
                    wall_s_before_save=config["dataset_wall_time_s_before_save"],
                    memory_bank_rows=sum(
                        b["memory_bank_shape"][0] for b in config["branches"]
                    ),
                )
            )
            if mode == "dual":
                gate = config["variants"]["gate_selected"]["adaptive_diagnostics"][
                    "gate"
                ]
                gates.append(
                    dict(
                        category=category,
                        seed=seed,
                        use_shrinkage=gate["use_shrinkage"],
                        group_consistency=gate["group_consistency"],
                        layer2_sensitivity_weight=gate["layer2_weight"],
                        layer3_stability=gate["layer3_relative_stability"],
                        final_layer2_pixel_weight=config["pixel_weights"][0],
                    )
                )
        if manifests[0] != manifests[1]:
            raise ValueError(
                "Feature and dual branch runs do not share a dataset split."
            )
        if set(manifests[0]["train_files"]) & set(manifests[0]["calibration_files"]):
            raise ValueError("Calibration data leaked into memory bank training.")
    expected = len(args.categories) * len(args.seeds) * 5
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} metric rows, got {len(rows)}")
    write_csv(args.output / "new_metrics.csv", rows)
    write_csv(args.output / "new_timings.csv", timings)
    write_csv(args.output / "new_gate_decisions.csv", gates)
    summaries, deltas, stats = paired_statistics(rows, "fixed_score")
    write_csv(args.output / "new_summary.csv", summaries)
    write_csv(args.output / "new_paired.csv", deltas)
    write_csv(args.output / "new_statistics.csv", stats)
    _, original_deltas, original_stats = paired_statistics(rows, "feature_original")
    write_csv(args.output / "new_vs_original_paired.csv", original_deltas)
    write_csv(args.output / "new_vs_original_statistics.csv", original_stats)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("E:/Datasets/MVTecAD/mvtec"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/restoration_20261022"
    )
    parser.add_argument(
        "--categories", nargs="+", default=["bottle", "pill", "transistor"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--historical-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if ROOT / "results" not in args.output.parents:
        raise ValueError("Output must be a named subdirectory under project results/.")
    args.output.mkdir(parents=True, exist_ok=True)
    historical = audit_history(args.output)
    print(
        "Historical gate reconstruction: 75/75 decisions and selected metrics verified.",
        flush=True,
    )
    if args.historical_only:
        print(json.dumps(historical, indent=2))
        return
    if args.analyze_only:
        summarize_new(args)
        return
    sources = sorted((ROOT / "src/patchcore").rglob("*.py")) + [
        ROOT / "bin/run_patchcore.py",
        Path(__file__),
    ]
    manifest = dict(
        categories=args.categories,
        seeds=args.seeds,
        train_fraction=0.8,
        normalization="evaluation_set_minmax",
        adaptive_samples=64,
        shrinkage=0.25,
        gate_mode="legacy",
        gate_thresholds=[0.8, 0.8, 0.9],
        num_workers=0,
        batch_size=2,
        note="New frozen protocol; historical exact split settings unavailable.",
        sources_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != manifest:
        raise ValueError(
            "Protocol/source changed. Use a different output directory; do not mix runs."
        )
    protocol_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    import torch
    import torchvision

    runtime = dict(
        python=sys.version,
        platform=platform.platform(),
        torch=torch.__version__,
        torchvision=torchvision.__version__,
        cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0),
    )
    (args.output / "runtime.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONUNBUFFERED"] = "1"
    total = len(args.categories) * len(args.seeds) * 2
    for i, (category, seed, mode) in enumerate(
        itertools.product(args.categories, args.seeds, ("feature", "dual")), 1
    ):
        group, cmd = command(category, seed, mode, args)
        receipt = args.output / group / "completed.json"
        if receipt.exists():
            print(f"[{i}/{total}] verified completed run: {group}", flush=True)
            continue
        if (args.output / group).exists():
            raise RuntimeError(
                f"Incomplete run {group}; inspect its log before retrying in a new directory."
            )
        print(f"[{i}/{total}] START {group}", flush=True)
        started = time.perf_counter()
        with (args.output / (group + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode:
            raise RuntimeError(
                f"Run {group} failed with exit code {result.returncode}; see its .log"
            )
        (args.output / (group + ".log")).rename(args.output / group / "run.log")
        receipt.write_text(
            json.dumps(
                dict(
                    command=cmd,
                    returncode=0,
                    process_wall_time_s=time.perf_counter() - started,
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[{i}/{total}] DONE {group} ({time.perf_counter()-started:.1f}s)",
            flush=True,
        )
    summarize_new(args)


if __name__ == "__main__":
    main()
