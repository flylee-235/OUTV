import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from torch.utils.data import Dataset

from .sam3_memory import SAM3Memory


@dataclass
class AnchorTargetSample:
    anchor_box: Tuple[int, int, int, int]
    target_box: Tuple[int, int, int, int]
    relative_position: torch.Tensor


def _sample_frame_indices(length: int, sample_n_frames: int, stride: int, random_time: bool) -> List[int]:
    base = list(range(0, length, max(1, stride)))
    if not base:
        base = [0]
    if len(base) >= sample_n_frames:
        start = random.randint(0, len(base) - sample_n_frames) if random_time else 0
        return base[start : start + sample_n_frames]
    return base + [base[-1]] * (sample_n_frames - len(base))


def _resize_video(video: torch.Tensor, sample_size: Tuple[int, int]) -> torch.Tensor:
    f, c, _, _ = video.shape
    return F.interpolate(video, size=sample_size, mode="bilinear", align_corners=False).view(
        f, c, sample_size[0], sample_size[1]
    )


def sample_anchor_target(
    pixel_values: torch.Tensor,
    target_size: Tuple[int, int],
    anchor_size: Tuple[int, int],
    overlap_ratio: Sequence[float],
    dynamic_anchor_size: bool,
) -> Dict[str, torch.Tensor]:
    if pixel_values.ndim != 4:
        raise ValueError(f"Expected FCHW pixel values, got {tuple(pixel_values.shape)}")
    f, c, h, w = pixel_values.shape
    th, tw = target_size
    ah, aw = anchor_size
    if dynamic_anchor_size:
        # Interpret anchor_size as a fixed minimum in this first implementation.
        ah = min(ah, h)
        aw = min(aw, w)
    if th > h or tw > w or ah > h or aw > w:
        raise ValueError(
            f"sample_size {(h, w)} must contain target {(th, tw)} and anchor {(ah, aw)}"
        )

    min_oh, max_oh, min_ow, max_ow = overlap_ratio
    oh = random.uniform(min_oh, max_oh)
    ow = random.uniform(min_ow, max_ow)
    offset_h = int((1.0 - oh) * th)
    offset_w = int((1.0 - ow) * tw)

    anchor_cy, anchor_cx = h // 2, w // 2
    target_cy = anchor_cy + offset_h if random.random() > 0.5 else anchor_cy - offset_h
    target_cx = anchor_cx + offset_w if random.random() > 0.5 else anchor_cx - offset_w

    ah0 = max(0, min(h - ah, anchor_cy - ah // 2))
    aw0 = max(0, min(w - aw, anchor_cx - aw // 2))
    th0 = max(0, min(h - th, target_cy - th // 2))
    tw0 = max(0, min(w - tw, target_cx - tw // 2))

    anchor = pixel_values[:, :, ah0 : ah0 + ah, aw0 : aw0 + aw]
    target = pixel_values[:, :, th0 : th0 + th, tw0 : tw0 + tw]
    mask = torch.ones((f, 1, th, tw), dtype=pixel_values.dtype)

    known_top = max(ah0, th0) - th0
    known_left = max(aw0, tw0) - tw0
    known_bottom = max(min(ah0 + ah, th0 + th) - th0, 0)
    known_right = max(min(aw0 + aw, tw0 + tw) - tw0, 0)
    if known_bottom > known_top and known_right > known_left:
        mask[:, :, known_top:known_bottom, known_left:known_right] = 0

    relative = torch.tensor(
        [
            th0 + th // 2 - (ah0 + ah // 2),
            tw0 + tw // 2 - (aw0 + aw // 2),
            ah,
            aw,
            th,
            tw,
        ],
        dtype=torch.float32,
    )
    return {
        "anchor_pixels_values": anchor,
        "target_pixels_values": target,
        "mask": mask,
        "relative_position": relative,
        "anchor_box": torch.tensor([aw0, ah0, aw0 + aw, ah0 + ah], dtype=torch.long),
        "target_box": torch.tensor([tw0, th0, tw0 + tw, th0 + th], dtype=torch.long),
    }


class SingleVideoIntrinsicDataset(Dataset):
    def __init__(
        self,
        video_path: str,
        sam3_root: Optional[str] = None,
        sample_size: Sequence[int] = (768, 768),
        sample_stride: int = 1,
        sample_n_frames: int = 64,
        target_size: Sequence[int] = (512, 512),
        anchor_size: Sequence[int] = (512, 512),
        overlap_ratio: Sequence[float] = (0.1, 1.0, 0.1, 1.0),
        dynamic_anchor_size: bool = False,
        random_sample_time: bool = True,
        prompt: str = "",
        length_multiplier: int = 1024,
    ):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        self.reader = VideoReader(str(self.video_path))
        self.video_length = len(self.reader)
        self.sam3 = SAM3Memory(sam3_root) if sam3_root else None
        self.sample_size = tuple(int(v) for v in sample_size)
        self.sample_stride = int(sample_stride)
        self.sample_n_frames = int(sample_n_frames)
        self.target_size = tuple(int(v) for v in target_size)
        self.anchor_size = tuple(int(v) for v in anchor_size)
        self.overlap_ratio = tuple(float(v) for v in overlap_ratio)
        self.dynamic_anchor_size = bool(dynamic_anchor_size)
        self.random_sample_time = bool(random_sample_time)
        self.prompt = prompt
        self.length_multiplier = int(length_multiplier)
        self.normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self) -> int:
        return max(1, self.length_multiplier)

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        frame_indices = _sample_frame_indices(
            self.video_length, self.sample_n_frames, self.sample_stride, self.random_sample_time
        )
        frames = self.reader.get_batch(frame_indices)
        pixel_values = torch.from_numpy(frames.asnumpy()).permute(0, 3, 1, 2).float() / 255.0
        pixel_values = _resize_video(pixel_values, self.sample_size)
        pixel_values = self.normalize(pixel_values)

        sampled = sample_anchor_target(
            pixel_values=pixel_values,
            target_size=self.target_size,
            anchor_size=self.anchor_size,
            overlap_ratio=self.overlap_ratio,
            dynamic_anchor_size=self.dynamic_anchor_size,
        )
        sample: Dict[str, torch.Tensor] = {
            "pixel_values": pixel_values,
            "fps": torch.tensor(self.sample_stride, dtype=torch.long),
            "video_length": torch.tensor(self.video_length, dtype=torch.long),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "relative_position": sampled["relative_position"],
            "anchor_box": sampled["anchor_box"],
            "target_box": sampled["target_box"],
            "anchor_pixels_values": sampled["anchor_pixels_values"],
            "target_pixels_values": sampled["target_pixels_values"],
            "mask": sampled["mask"],
            "text": self.prompt,
            "ori_text": self.prompt,
        }
        if self.sam3 is not None:
            memory = self.sam3.load_clip(frame_indices, mask_size=self.sample_size)
            sample["sam3_dense_embeddings"] = memory["embeddings"]
            sample["sam3_class_masks"] = memory["class_masks"]
        return sample

