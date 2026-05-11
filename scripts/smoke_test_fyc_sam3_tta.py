import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fyc_sam3_tta.sam3_memory import SAM3Memory
from fyc_sam3_tta.single_video_dataset import SingleVideoIntrinsicDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--sam3-root", required=True)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()

    memory = SAM3Memory(args.sam3_root)
    print(json.dumps(memory.validate(), ensure_ascii=False, indent=2))

    dataset = SingleVideoIntrinsicDataset(
        video_path=args.video_path,
        sam3_root=args.sam3_root,
        sample_size=(768, 768),
        sample_stride=1,
        sample_n_frames=64,
        target_size=(512, 512),
        anchor_size=(512, 512),
        random_sample_time=False,
        length_multiplier=args.samples,
    )
    for i in range(args.samples):
        sample = dataset[i]
        report = {
            "idx": i,
            "pixel_values": tuple(sample["pixel_values"].shape),
            "anchor": tuple(sample["anchor_pixels_values"].shape),
            "target": tuple(sample["target_pixels_values"].shape),
            "mask_minmax": (float(sample["mask"].min()), float(sample["mask"].max())),
            "relative_position": sample["relative_position"].tolist(),
            "anchor_box": sample["anchor_box"].tolist(),
            "target_box": sample["target_box"].tolist(),
            "sam3_dense": tuple(sample["sam3_dense_embeddings"].shape),
            "sam3_class_masks": tuple(sample["sam3_class_masks"].shape),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        assert torch.isfinite(sample["pixel_values"]).all()
        assert sample["mask"].min() >= 0 and sample["mask"].max() <= 1


if __name__ == "__main__":
    main()
