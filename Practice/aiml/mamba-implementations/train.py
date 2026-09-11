"""Bounded CPU experiments. Run `python train.py --help` for the small CLI."""
import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from mamba_minimal.model import CopyModel
from mamba_minimal.task import recall_metrics, selective_copy

KINDS = ("mamba1", "mamba2", "mamba3", "mamba1_constant")
PRESETS = {
    "smoke": dict(width=32, state_size=8, layers=1, length=32, copies=3, batch_size=4, steps=5, lr=0.003, eval_batches=2),
    "bounded": dict(width=32, state_size=8, layers=1, length=32, copies=3, batch_size=16, steps=300, lr=0.003, eval_batches=16),
    "overfit": dict(width=32, state_size=8, layers=1, length=16, copies=2, batch_size=4, steps=400, lr=0.01, eval_batches=2),
}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_model(kind, cfg):
    return CopyModel(kind, **{k: cfg[k] for k in ("width", "state_size", "layers", "classes")})


def batch(cfg, generator, device, length=None):
    x, target = selective_copy(cfg["batch_size"], length or cfg["length"], generator,
                               copies=cfg["copies"], classes=cfg["classes"])
    return x.to(device), target.to(device)


def update(model, optimizer, data):
    optimizer.zero_grad(set_to_none=True)
    loss, acc = recall_metrics(model(data[0]), data[1])
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite loss")
    loss.backward()
    # error_if_nonfinite checks all gradient norms before clipping.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), acc.item(), norm.item()


@torch.no_grad()
def evaluate(model, cfg, device, seed, length):
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses, accuracies = [], []
    for _ in range(cfg["eval_batches"]):
        x, targets = batch(cfg, generator, device, length)
        loss, accuracy = recall_metrics(model(x), targets)
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    return dict(length=length, seed=seed, examples=cfg["batch_size"]*cfg["eval_batches"],
                recall_tokens=cfg["batch_size"]*cfg["eval_batches"]*cfg["copies"],
                loss=sum(losses)/len(losses), accuracy=sum(accuracies)/len(accuracies))


def benchmark(kind, cfg, device):
    torch.manual_seed(cfg["seed"])
    model = make_model(kind, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
    generator = torch.Generator().manual_seed(cfg["train_seed"])
    data = batch(cfg, generator, device)
    for _ in range(2):
        update(model, optimizer, data)
    synchronize(device)
    started = time.perf_counter()
    for _ in range(5):
        update(model, optimizer, data)
    synchronize(device)
    return dict(model=kind, parameters=sum(p.numel() for p in model.parameters()),
                measured_steps=5, warmup_steps=2, seconds_per_step=(time.perf_counter()-started)/5)


def metadata(cfg, device):
    return dict(config=cfg, device=str(device), device_name=(torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()),
                python=platform.python_version(), torch=str(torch.__version__), platform=platform.platform(),
                threads=torch.get_num_threads(), created_utc=datetime.now(timezone.utc).isoformat())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def fit(kind, cfg, device, output):
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    model = make_model(kind, cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=0)
    generator = torch.Generator().manual_seed(cfg["train_seed"])
    fixed = batch(cfg, generator, device) if cfg["preset"] == "overfit" else None
    rows = []
    synchronize(device)
    started = time.perf_counter()
    model.train()
    for step in range(1, cfg["steps"]+1):
        loss, acc, norm = update(model, optimizer, fixed if fixed is not None else batch(cfg, generator, device))
        rows.append(dict(step=step, loss=loss, recall_accuracy=acc, gradient_norm=norm))
        if step == 1 or step % 50 == 0 or step == cfg["steps"]:
            print(f"{kind} {step}/{cfg['steps']}: loss={loss:.4f}, recall={acc:.3f}", flush=True)
    synchronize(device)
    train_seconds = time.perf_counter() - started
    result = metadata(cfg, device)
    result.update(model=kind, parameters=sum(p.numel() for p in model.parameters()),
                  train_seconds=train_seconds, final_training_loss=rows[-1]["loss"],
                  mean_last_20_training_loss=sum(r["loss"] for r in rows[-20:])/len(rows[-20:]))
    if fixed is not None:
        with torch.no_grad():
            loss, accuracy = recall_metrics(model(fixed[0]), fixed[1])
        result["fixed_batch"] = dict(loss=loss.item(), accuracy=accuracy.item(),
                                      passed=loss.item() < 0.05 and accuracy.item() == 1.0)
    result["held_out"] = [evaluate(model, cfg, device, cfg["eval_seed"], cfg["length"]),
                          evaluate(model, cfg, device, cfg["long_eval_seed"], 2*cfg["length"])]
    synchronize(device)
    result["elapsed_seconds"] = time.perf_counter() - started
    with (output / "training.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    # A final model checkpoint, sufficient for reload/evaluation, not optimizer resume.
    torch.save(dict(model=kind, config=cfg, state_dict=model.cpu().state_dict()), output / "checkpoint.pt")
    write_json(output / "result.json", result)
    if fixed is not None and not result["fixed_batch"]["passed"]:
        raise AssertionError(f"{kind} did not overfit: {result['fixed_batch']}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("benchmark", "train", "evaluate"))
    parser.add_argument("--model", choices=(*KINDS, "all"), default="all")
    parser.add_argument("--preset", choices=PRESETS, default="smoke")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/smoke"))
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error("threads and steps must be positive")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        if args.checkpoint is None:
            parser.error("evaluate requires --checkpoint")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cfg = saved["config"]
        model = make_model(saved["model"], cfg).to(device)
        model.load_state_dict(saved["state_dict"])
        result = metadata(cfg, device)
        result.update(model=saved["model"], checkpoint=str(args.checkpoint), held_out=[
            evaluate(model, cfg, device, cfg["eval_seed"], cfg["length"]),
            evaluate(model, cfg, device, cfg["long_eval_seed"], 2*cfg["length"])])
        write_json(args.output / "evaluation.json", result)
        print(json.dumps(result["held_out"], indent=2))
        return
    cfg = dict(PRESETS[args.preset], preset=args.preset, classes=4, seed=args.seed,
               train_seed=args.seed+1000, eval_seed=args.seed+2000, long_eval_seed=args.seed+3000,
               optimizer="Adam", weight_decay=0, clip_norm=1.0, dtype="float32")
    if args.steps is not None:
        cfg["steps"] = args.steps
    kinds = KINDS if args.model == "all" else (args.model,)
    if args.mode == "benchmark":
        result = metadata(cfg, device)
        result["measurements"] = [benchmark(kind, cfg, device) for kind in kinds]
        write_json(args.output / "benchmark.json", result)
        print(json.dumps(result["measurements"], indent=2))
    else:
        results = [fit(kind, cfg, device, args.output / kind) for kind in kinds]
        write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()
