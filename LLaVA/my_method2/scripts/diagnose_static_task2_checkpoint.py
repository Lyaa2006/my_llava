#!/usr/bin/env python3
"""Checkpoint diagnostics for my_method2 static task2 underfitting."""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


DEFAULT_ROOT = Path("/mnt/lyaa/my_llava")
DEFAULT_CKPT_ROOT = DEFAULT_ROOT / "checkpoint/UCIT/LLaVA-1.5"

LANG_LORA_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\."
    r"(?P<block>self_attn|mlp)\."
    r"(?P<proj>[^.]+)\."
    r"lora_(?P<side>[AB])\.lora[AB]\."
    r"(?P<expert>\d+)\.(?:mlp\.)?weight$"
)
LANG_ROUTER_RE = re.compile(
    r"model\.layers\.(?P<layer>\d+)\."
    r"(?P<block>self_attn|mlp)\."
    r"(?P<proj>[^.]+)\.lora_router\.weight$"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def load_adapter(ckpt_dir: Path) -> Dict[str, torch.Tensor]:
    path = ckpt_dir / "adapter_model.bin"
    if not path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {path}")
    return torch.load(path, map_location="cpu")


def tensor_norm(value: torch.Tensor) -> float:
    return float(value.detach().float().norm().item())


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: Iterable[float]) -> float:
    values = list(values)
    return float(statistics.median(values)) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round(q * (len(values) - 1)))))
    return float(values[idx])


def adapter_scale(ckpt_dir: Path) -> float:
    cfg = load_json(ckpt_dir / "adapter_config.json")
    r = float(cfg.get("r") or 1.0)
    alpha = float(cfg.get("lora_alpha") or 1.0)
    return alpha / r if r else 1.0


def extract_lora_entries(state: Dict[str, torch.Tensor]) -> dict:
    entries = {}
    for name, value in state.items():
        match = LANG_LORA_RE.search(name)
        if not match or not torch.is_tensor(value):
            continue
        info = match.groupdict()
        key = (
            int(info["layer"]),
            info["block"],
            info["proj"],
            int(info["expert"]),
            info["side"],
        )
        entries[key] = value
    return entries


def extract_router_entries(state: Dict[str, torch.Tensor]) -> dict:
    entries = {}
    for name, value in state.items():
        match = LANG_ROUTER_RE.search(name)
        if not match or not torch.is_tensor(value):
            continue
        info = match.groupdict()
        key = (int(info["layer"]), info["block"], info["proj"])
        entries[key] = value
    return entries


def pair_strengths(entries: dict, scale: float) -> Dict[int, List[float]]:
    pairs = defaultdict(dict)
    for layer, block, proj, expert, side in entries:
        pairs[(layer, block, proj, expert)][side] = entries[(layer, block, proj, expert, side)]

    by_expert = defaultdict(list)
    for (_layer, _block, _proj, expert), sides in pairs.items():
        a = sides.get("A")
        b = sides.get("B")
        if a is None or b is None:
            continue
        # Upper bound proxy for ||B @ A||. It is fast enough for all checkpoints.
        by_expert[expert].append(tensor_norm(a) * tensor_norm(b) * scale)
    return by_expert


def summarize_adapter(label: str, ckpt_dir: Path) -> dict:
    state = load_adapter(ckpt_dir)
    cfg = load_json(ckpt_dir / "adapter_config.json")
    scale = adapter_scale(ckpt_dir)
    entries = extract_lora_entries(state)
    routers = extract_router_entries(state)
    strengths = pair_strengths(entries, scale)

    side_norms = defaultdict(lambda: defaultdict(list))
    for (_layer, _block, _proj, expert, side), value in entries.items():
        side_norms[expert][side].append(tensor_norm(value))

    experts = sorted(set(side_norms) | set(strengths))
    expert_rows = []
    for expert in experts:
        b_norms = side_norms[expert].get("B", [])
        proxy = strengths.get(expert, [])
        expert_rows.append(
            {
                "expert": expert,
                "a_mean": safe_mean(side_norms[expert].get("A", [])),
                "b_mean": safe_mean(b_norms),
                "b_active": sum(1 for value in b_norms if value > 1e-8),
                "pairs": len(proxy),
                "proxy_mean": safe_mean(proxy),
                "proxy_p95": percentile(proxy, 0.95),
                "proxy_max": max(proxy) if proxy else 0.0,
            }
        )

    router_norms = [tensor_norm(value) for value in routers.values()]
    return {
        "label": label,
        "path": str(ckpt_dir),
        "config": cfg,
        "state": state,
        "entries": entries,
        "routers": routers,
        "scale": scale,
        "expert_rows": expert_rows,
        "router_count": len(router_norms),
        "router_norm_mean": safe_mean(router_norms),
    }


def compare_entries(before: dict, after: dict) -> List[dict]:
    common = sorted(set(before) & set(after))
    rows = defaultdict(lambda: {"total": 0, "changed": 0, "zero_to_nonzero": 0, "diffs": [], "ratios": [], "top": []})
    for key in common:
        expert = key[3]
        old = before[key].detach().float()
        new = after[key].detach().float()
        diff = new - old
        diff_norm = tensor_norm(diff)
        old_norm = tensor_norm(old)
        new_norm = tensor_norm(new)

        row = rows[expert]
        row["total"] += 1
        row["diffs"].append(diff_norm)
        if diff_norm > 1e-8:
            row["changed"] += 1
        if old_norm <= 1e-8 and new_norm > 1e-8:
            row["zero_to_nonzero"] += 1
        row["ratios"].append(diff_norm / max(new_norm, 1e-12))
        if diff_norm > 1e-8:
            row["top"].append((diff_norm, key, old_norm, new_norm))

    result = []
    for expert in sorted(rows):
        row = rows[expert]
        top = sorted(row["top"], reverse=True)[:5]
        result.append(
            {
                "expert": expert,
                "changed": row["changed"],
                "total": row["total"],
                "zero_to_nonzero": row["zero_to_nonzero"],
                "diff_mean": safe_mean(row["diffs"]),
                "diff_p95": percentile(row["diffs"], 0.95),
                "diff_max": max(row["diffs"]) if row["diffs"] else 0.0,
                "diff_over_after_mean": safe_mean(row["ratios"]),
                "top": top,
            }
        )
    return result


def compare_routers(before: dict, after: dict) -> dict:
    common = sorted(set(before) & set(after))
    diffs = []
    for key in common:
        diffs.append(tensor_norm(after[key].detach().float() - before[key].detach().float()))
    return {
        "count": len(common),
        "changed": sum(1 for value in diffs if value > 1e-8),
        "diff_mean": safe_mean(diffs),
        "diff_max": max(diffs) if diffs else 0.0,
    }


def summarize_trainer_state(ckpt_dir: Path) -> Optional[dict]:
    state = load_json(ckpt_dir / "trainer_state.json")
    if not state:
        return None
    losses = [float(item["loss"]) for item in state.get("log_history", []) if "loss" in item]
    if not losses:
        return None
    return {
        "count": len(losses),
        "mean": safe_mean(losses),
        "median": safe_median(losses),
        "p95": percentile(losses, 0.95),
        "p99": percentile(losses, 0.99),
        "max": max(losses),
        "over_10": sum(1 for value in losses if value > 10),
        "over_100": sum(1 for value in losses if value > 100),
        "over_1000": sum(1 for value in losses if value > 1000),
    }


def summarize_cache(cache_dir: Optional[Path]) -> Optional[dict]:
    if cache_dir is None:
        return None
    meta = load_json(cache_dir / "meta.json")
    if not meta:
        return None
    return {
        "path": str(cache_dir),
        "teacher": meta.get("teacher"),
        "cache_source": meta.get("cache_source"),
        "snapshot_cur_task": meta.get("snapshot_cur_task"),
        "entries": meta.get("entries", meta.get("cached_entries")),
        "total": meta.get("total", meta.get("num_samples")),
        "hidden_layer": meta.get("description_hidden_layer"),
        "max_tokens": meta.get("description_max_tokens"),
    }


def print_adapter_summary(summary: dict):
    cfg = summary["config"]
    print(f"\n[{summary['label']}]")
    print(f"path: {summary['path']}")
    print(
        "config: "
        f"cur_task={cfg.get('cur_task')} expert_num={cfg.get('expert_num')} "
        f"r={cfg.get('r')} alpha={cfg.get('lora_alpha')} scale={summary['scale']:.4f}"
    )
    print(f"language_lora_tensors={len(summary['entries'])} router_tensors={summary['router_count']} router_norm_mean={summary['router_norm_mean']:.6f}")
    print("expert  A_mean    B_mean    B_active/pairs  proxy_mean  proxy_p95   proxy_max")
    for row in summary["expert_rows"]:
        print(
            f"{row['expert']:>6}  "
            f"{row['a_mean']:>8.4f}  {row['b_mean']:>8.4f}  "
            f"{row['b_active']:>4}/{row['pairs']:<4}      "
            f"{row['proxy_mean']:>8.4f}  {row['proxy_p95']:>8.4f}  {row['proxy_max']:>8.4f}"
        )


def print_compare(title: str, rows: List[dict]):
    print(f"\n[{title}]")
    print("expert  changed/total  zero_to_nonzero  diff_mean  diff_p95   diff_max   diff/after")
    for row in rows:
        print(
            f"{row['expert']:>6}  "
            f"{row['changed']:>4}/{row['total']:<4}      "
            f"{row['zero_to_nonzero']:>4}          "
            f"{row['diff_mean']:>8.4f}  {row['diff_p95']:>8.4f}  "
            f"{row['diff_max']:>8.4f}  {row['diff_over_after_mean']:>9.4f}"
        )
    top = []
    for row in rows:
        top.extend(row["top"])
    top = sorted(top, reverse=True)[:8]
    if top:
        print("top_changed_tensors:")
        for diff_norm, key, old_norm, new_norm in top:
            layer, block, proj, expert, side = key
            print(
                f"  layer={layer:02d} {block}.{proj} lora_{side} expert={expert} "
                f"diff={diff_norm:.4f} old={old_norm:.4f} new={new_norm:.4f}"
            )


def print_training_summary(label: str, summary: Optional[dict]):
    print(f"\n[{label} trainer_state]")
    if summary is None:
        print("No trainer_state loss history found.")
        return
    print(
        f"loss_count={summary['count']} mean={summary['mean']:.4f} "
        f"median={summary['median']:.4f} p95={summary['p95']:.4f} "
        f"p99={summary['p99']:.4f} max={summary['max']:.4f}"
    )
    print(
        f"loss_spikes: >10={summary['over_10']} "
        f">100={summary['over_100']} >1000={summary['over_1000']}"
    )


def print_cache_summary(summary: Optional[dict]):
    print("\n[description cache]")
    if summary is None:
        print("No cache meta found.")
        return
    print(
        f"path={summary['path']}\n"
        f"teacher={summary['teacher']} cache_source={summary['cache_source']} "
        f"snapshot_cur_task={summary['snapshot_cur_task']} "
        f"entries={summary['entries']} total={summary['total']} "
        f"hidden_layer={summary['hidden_layer']} max_tokens={summary['max_tokens']}"
    )


def print_takeaways(
    my_compare: List[dict],
    hide_compare: Optional[List[dict]],
    my_task2: dict,
):
    print("\n[diagnostic takeaways]")
    changed = {row["expert"]: row["changed"] for row in my_compare}
    changed_experts = [expert for expert, count in changed.items() if count > 0]
    if changed_experts == [1]:
        print("- my_method2 static task2 only changed expert 1; expert 0 is preserved exactly and experts 2+ remain unused/random.")
    else:
        print(f"- my_method2 changed experts: {changed_experts}")
    cfg = my_task2["config"]
    cur_task = cfg.get("cur_task")
    if cur_task is not None:
        cur_task = int(cur_task)
        proxy_by_expert = {row["expert"]: row["proxy_mean"] for row in my_task2["expert_rows"]}
        old_proxy = proxy_by_expert.get(cur_task - 1, 0.0)
        cur_proxy = proxy_by_expert.get(cur_task, 0.0)
        if cur_task > 0 and cur_proxy > 0:
            print(
                f"- effective LoRA strength proxy old/current = "
                f"{old_proxy:.4f}/{cur_proxy:.4f} = {old_proxy / cur_proxy:.2f}x."
            )
    if hide_compare:
        hide_changed = [row["expert"] for row in hide_compare if row["changed"] > 0]
        print(f"- HiDe comparison changed experts: {hide_changed}")
    print("- In this HiDe-style forward path, training uses only cur_task expert, while inference fuses old+current experts in most layers.")
    print("- If task2 accuracy is low while task1 is high, the checkpoint should be read as strong retention plus weak/current-task adaptation or old-task dominance.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task1",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "my_method2/Task1_llava_lora",
        help="my_method2 task1 checkpoint used before task2.",
    )
    parser.add_argument(
        "--task2",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "my_method2/static/Task2_llava_lora",
        help="my_method2 static task2 checkpoint.",
    )
    parser.add_argument(
        "--prev-task",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "my_method2/static/Task2_llava_lora/prev_task",
        help="Frozen previous-task adapter snapshot saved under the static task2 checkpoint.",
    )
    parser.add_argument(
        "--hide-task1",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "HiDe/Task1_llava_lora",
        help="Optional HiDe task1 checkpoint for contrast.",
    )
    parser.add_argument(
        "--hide-task2",
        type=Path,
        default=DEFAULT_CKPT_ROOT / "HiDe/Task2_llava_lora",
        help="Optional HiDe task2 checkpoint for contrast.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_ROOT / "checkpoint/UCIT/full_my_method2/static/static_20260627_230808/desc_cache_task2",
        help="Static description cache directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    my_t1 = summarize_adapter("my_method2 task1", args.task1)
    my_t2 = summarize_adapter("my_method2 static task2", args.task2)

    print_training_summary("my_method2 static task2", summarize_trainer_state(args.task2))
    print_cache_summary(summarize_cache(args.cache_dir))
    print_adapter_summary(my_t1)
    print_adapter_summary(my_t2)

    my_compare = compare_entries(my_t1["entries"], my_t2["entries"])
    print_compare("my_method2 task1 -> static task2", my_compare)
    if args.prev_task.exists():
        prev_task = summarize_adapter("my_method2 static prev_task snapshot", args.prev_task)
        prev_compare = compare_entries(my_t1["entries"], prev_task["entries"])
        print_compare("my_method2 task1 -> static prev_task snapshot", prev_compare)
    router_compare = compare_routers(my_t1["routers"], my_t2["routers"])
    print(
        "\n[my_method2 router task1 -> static task2]\n"
        f"common={router_compare['count']} changed={router_compare['changed']} "
        f"diff_mean={router_compare['diff_mean']:.6f} diff_max={router_compare['diff_max']:.6f}"
    )

    hide_compare = None
    if args.hide_task1.exists() and args.hide_task2.exists():
        hide_t1 = summarize_adapter("HiDe task1", args.hide_task1)
        hide_t2 = summarize_adapter("HiDe task2", args.hide_task2)
        print_adapter_summary(hide_t2)
        hide_compare = compare_entries(hide_t1["entries"], hide_t2["entries"])
        print_compare("HiDe task1 -> task2", hide_compare)

    print_takeaways(my_compare, hide_compare, my_t2)


if __name__ == "__main__":
    main()
