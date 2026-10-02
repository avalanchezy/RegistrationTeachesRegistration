"""Synthetic CPU/CUDA pipeline check; no real-data accuracy claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.engine import TrainingConfig, train
from task2reg.journal.inference import predict
from task2reg.journal.evaluation import evaluate_records
from task2reg.journal.pseudo import export_verified_pseudo
from task2reg.journal.runtime import write_json, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty smoke output directory")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    rng = np.random.default_rng(12)
    points = rng.normal(size=(96, 3))
    points = points / np.linalg.norm(points, axis=1, keepdims=True) * 2 + 7.5
    records = []
    for index, split in enumerate(("train", "val", "unlabeled")):
        affine = np.eye(4)
        coordinates = np.stack(np.meshgrid(*([np.arange(16)] * 3), indexing="ij"), axis=-1)
        radius = np.linalg.norm(coordinates - 7.5, axis=-1)
        # A constant HU500 volume normalizes to zero; repeated GroupNorm then
        # amplifies bias gradients pathologically. Exercise AMP on a real
        # spatial signal, including high-density shell, air and acquisition noise.
        image = np.rint(-500 + 2200 * np.exp(-((radius - 3) / .9) ** 2)
                        + rng.normal(0, 30, radius.shape) + index).astype(np.int16)
        candidates = np.repeat(np.eye(4)[None], 3, axis=0)
        candidates[1, 0, 3] = 1.5
        candidates[2, 1, 3] = -2.
        case = dict(image=image, affine=affine, points=points.astype(np.float32),
                    anchors=points.astype(np.float32), candidates=candidates)
        if split != "unlabeled":
            labels = np.zeros(image.shape, dtype=np.uint8)
            labels[5:11, 5:11, 5:11] = 1
            case.update(label=labels, transform=np.eye(4))
        np.savez_compressed(output / f"{split}.npz", **case)
        records.append(dict(source="synthetic", case_id=f"synthetic:{index}", patient_id=f"s{index}",
                            jaw="upper", split=split, npz_path=f"{split}.npz",
                            reference_kind="none" if split == "unlabeled" else "manual",
                            candidate_provenance={"kind": "geometry_only"}))
    manifest = output / "manifest.json"
    write_json(manifest, {"schema_version": 1, "records": records})
    config = TrainingConfig.from_json(Path(__file__).resolve().parents[1] / "configs/journal/smoke.json")
    config = replace(config, device=args.device, amp=args.amp)
    train(manifest, output / "training", replace(config, epochs=1))
    checkpoint = output / "training/last.pt"
    first_state = load_checkpoint(checkpoint)["model"]
    training_result = train(manifest, output / "training", config, resume=checkpoint)
    final_state = load_checkpoint(checkpoint)["model"]
    weights_changed = any(not torch.equal(first_state[name], value) for name, value in final_state.items())
    if training_result["global_step"] == 0 or not weights_changed:
        raise RuntimeError("smoke did not perform effective optimizer updates; inspect AMP/gradient logs")
    predict(manifest, checkpoint, output / "predictions", split="val", device=args.device,
            refinement_steps=2, point_budget=64, evaluate=True)
    evaluation = json.loads((output / "predictions/evaluation_input.json").read_text())
    report = evaluate_records(evaluation["records"], bootstrap_samples=32)
    write_json(output / "evaluation.json", report)
    export_verified_pseudo(manifest, checkpoint, output / "pseudo", device=args.device,
                           starts=4, sectors=4, refinement_steps=1, point_budget=64)
    status = {"smoke_status": "complete", "device": args.device, "amp": args.amp,
              "synthetic_only": True, "artifacts": str(output),
              "effective_updates": training_result["global_step"], "weights_changed": weights_changed}
    write_json(output / "smoke_result.json", status)
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
