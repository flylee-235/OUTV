import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SAM3Memory:
    """Reader for precomputed SAM3 dense embeddings and class masks."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.metadata_path = self.root / "metadata.json"
        self.embedding_dir = self.root / "image_embeddings"
        self.mask_dir = self.root / "class_masks"
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"SAM3 metadata not found: {self.metadata_path}")
        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata: Dict = json.load(f)
        self.categories: List[str] = list(self.metadata.get("categories", []))
        self.frame_count = int(self.metadata.get("frame_count", 0))

    def validate(self) -> Dict[str, object]:
        embedding_files = sorted(self.embedding_dir.glob("*.npz"))
        mask_files = sorted(self.mask_dir.glob("*.npz"))
        return {
            "root": str(self.root),
            "frame_count": self.frame_count,
            "categories": self.categories,
            "embedding_count": len(embedding_files),
            "mask_count": len(mask_files),
            "has_all_embeddings": len(embedding_files) >= self.frame_count,
            "has_all_masks": len(mask_files) >= self.frame_count,
        }

    def load_embedding(self, frame_idx: int) -> torch.Tensor:
        path = self.embedding_dir / f"{frame_idx:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"SAM3 embedding not found: {path}")
        with np.load(path) as data:
            if "embedding" not in data:
                raise KeyError(f"{path} does not contain key 'embedding'")
            arr = data["embedding"]
        if arr.ndim != 3:
            raise ValueError(f"Expected embedding (C,H,W), got {arr.shape} from {path}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def load_class_masks(self, frame_idx: int) -> torch.Tensor:
        path = self.mask_dir / f"{frame_idx:06d}.npz"
        if not path.exists():
            raise FileNotFoundError(f"SAM3 class masks not found: {path}")
        with np.load(path) as data:
            if "masks" not in data:
                raise KeyError(f"{path} does not contain key 'masks'")
            arr = data["masks"]
        if arr.ndim != 3:
            raise ValueError(f"Expected masks (K,H,W), got {arr.shape} from {path}")
        return torch.from_numpy(arr.astype(np.float32, copy=False))

    def load_clip(
        self,
        frame_indices: Sequence[int],
        mask_size: Optional[Tuple[int, int]] = None,
        embedding_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        embeddings = torch.stack([self.load_embedding(i) for i in frame_indices], dim=0)
        masks = torch.stack([self.load_class_masks(i) for i in frame_indices], dim=0)
        if embedding_size is not None:
            embeddings = F.interpolate(
                embeddings, size=embedding_size, mode="bilinear", align_corners=False
            )
        if mask_size is not None:
            masks = F.interpolate(masks, size=mask_size, mode="nearest")
        return {"embeddings": embeddings, "class_masks": masks}


def crop_feature_by_box(
    feature: torch.Tensor,
    box_xyxy: Sequence[int],
    image_size: Tuple[int, int],
    output_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """Crop a CHW/FCHW feature using a pixel-space box from an image."""
    if feature.ndim == 3:
        feature_in = feature.unsqueeze(0)
        squeeze = True
    elif feature.ndim == 4:
        feature_in = feature
        squeeze = False
    else:
        raise ValueError(f"Expected CHW or FCHW feature, got shape {tuple(feature.shape)}")

    _, _, fh, fw = feature_in.shape
    ih, iw = image_size
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    fx0 = max(0, min(fw - 1, round(x0 / iw * fw)))
    fx1 = max(fx0 + 1, min(fw, round(x1 / iw * fw)))
    fy0 = max(0, min(fh - 1, round(y0 / ih * fh)))
    fy1 = max(fy0 + 1, min(fh, round(y1 / ih * fh)))
    cropped = feature_in[:, :, fy0:fy1, fx0:fx1]
    if output_size is not None:
        cropped = F.interpolate(cropped, size=output_size, mode="bilinear", align_corners=False)
    return cropped.squeeze(0) if squeeze else cropped

