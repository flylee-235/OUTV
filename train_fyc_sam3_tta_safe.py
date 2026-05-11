import argparse
import datetime
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def _refuse_shell_delete(command: str) -> int:
    raise RuntimeError(f"Refusing shell command from safe train wrapper: {command}")


def _refuse_remove(path: str) -> None:
    raise RuntimeError(f"Refusing checkpoint deletion from safe train wrapper: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--launcher", choices=["pytorch", "slurm"], default="pytorch")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config = OmegaConf.load(args.config)
    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = Path(config.get("output_dir", "outputs/fyc_sam3_tta/tta"))
    run_output = output_base / run_id
    if run_output.exists():
        raise FileExistsError(f"Refusing to reuse existing output dir: {run_output}")

    print({"output_dir": str(run_output), "config": args.config, "launcher": args.launcher})
    if args.dry_run:
        return

    run_output.mkdir(parents=True, exist_ok=False)
    config.output_dir = str(run_output)
    effective_config = run_output / "effective_config.yaml"
    OmegaConf.save(config, effective_config)

    spec = importlib.util.spec_from_file_location("fyc_train_original", repo_root / "train.py")
    train_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_module)
    train_module.os.system = _refuse_shell_delete
    train_module.os.remove = _refuse_remove
    train_module.main(
        name=Path(args.config).stem,
        launcher=args.launcher,
        use_wandb=args.wandb,
        **OmegaConf.load(effective_config),
    )


if __name__ == "__main__":
    main()

