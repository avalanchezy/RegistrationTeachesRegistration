"""Audit a verifier with hidden manual references, then freeze its development policy."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from task2reg.journal.verification import VerificationConfig
from task2reg.journal.verification_audit import audit_verifier, freeze_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collect = commands.add_parser("collect", help="references are only used after verification decisions")
    for name in ("manifest", "checkpoint", "gate-config", "output-dir"):
        collect.add_argument("--" + name, type=Path, required=True)
    collect.add_argument("--data-root", type=Path)
    collect.add_argument("--device", default="cuda")
    collect.add_argument("--split", choices=("val", "test", "external_test"), default="val")
    collect.add_argument("--purpose", choices=("development", "frozen"), default="development")
    collect.add_argument("--policy", type=Path, dest="policy_path")
    collect.add_argument("--source")
    for name, default in (("starts", 8), ("sectors", 4), ("refinement-steps", 20), ("point-budget", 4096), ("seed", 0)):
        collect.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", .25), ("success-threshold-mm", 2.), ("alpha", .05), ("risk-target", .05)):
        collect.add_argument("--" + name, type=float, default=default)
    freeze = commands.add_parser("freeze", help="freeze selected settings and declare all development exposure")
    freeze.add_argument("--selected-audit", type=Path, required=True)
    freeze.add_argument("--additional-development-audit", type=Path, action="append", default=[])
    freeze.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    if args.pop("command") == "freeze":
        result = freeze_policy(args["selected_audit"], args["output"],
                               additional_development_audits=args["additional_development_audit"])
        print(json.dumps({"policy_sha256": result["policy_sha256"], "output": str(args["output"])}))
    else:
        args["config"] = VerificationConfig(**json.loads(args.pop("gate_config").read_text()))
        result = audit_verifier(**args)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
