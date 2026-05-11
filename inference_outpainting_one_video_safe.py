import argparse
import datetime
import importlib.util
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf


DEFAULT_VIDEO = "/home/610-ltf/DL/Datasets/data/video/摹阮郜女仙图卷.mp4"


def load_original_module(repo_root: Path):
    path = repo_root / "inference_outpainting-dir.py"
    spec = importlib.util.spec_from_file_location("fyc_original_inference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video-path", default=DEFAULT_VIDEO)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Target video not found: {video_path}")

    config = OmegaConf.load(args.config)
    if args.preflight:
        checks = {
            "video_path": str(video_path),
            "video_exists": video_path.exists(),
            "pretrained_model_path": config.get("pretrained_model_path"),
            "pretrained_model_exists": Path(config.get("pretrained_model_path", "")).exists(),
            "motion_pretrained_model_path": config.get("motion_pretrained_model_path"),
            "motion_pretrained_model_exists": Path(config.get("motion_pretrained_model_path", "")).exists(),
            "lmm_path": config.get("lmm_path"),
            "lmm_path_exists": Path(config.get("lmm_path", "")).exists(),
            "image_pretrained_model_path": config.get("image_pretrained_model_path"),
            "image_pretrained_model_exists": Path(config.get("image_pretrained_model_path", "")).exists(),
        }
        import torch
        import xformers  # noqa: F401

        load_original_module(repo_root)
        checks["cuda_available"] = torch.cuda.is_available()
        checks["cuda_device_count"] = torch.cuda.device_count()
        print(checks)
        missing = [k for k, v in checks.items() if k.endswith("_exists") and not v]
        if missing:
            raise FileNotFoundError(f"Missing required paths: {missing}")
        if not checks["cuda_available"]:
            raise RuntimeError("CUDA is not available")
        return

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_base = Path(config.get("output_dir", "outputs/fyc_sam3_tta/zero_shot"))
    output_dir = output_base / run_id
    if output_dir.exists():
        raise FileExistsError(f"Refusing to reuse existing output dir: {output_dir}")

    one_video_dir = output_dir / "_input"
    planned = {
        "output_dir": str(output_dir),
        "one_video_dir": str(one_video_dir),
        "video_path": str(video_path),
        "config": args.config,
    }
    print(planned)
    if args.dry_run:
        return

    one_video_dir.mkdir(parents=True, exist_ok=False)
    link_path = one_video_dir / video_path.name
    link_path.symlink_to(video_path)

    config.output_dir = str(output_dir)
    config.video_dir = str(one_video_dir)
    temp_config = output_dir / "effective_config.yaml"
    OmegaConf.save(config, temp_config)

    cmd = [sys.executable, str(repo_root / "inference_outpainting-dir.py"), "--config", str(temp_config)]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
