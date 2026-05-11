# FollowYourCanvas + SAM3 水墨单视频外推研究上下文

## 当前主线判断

当前 SD3 单图先验路线已经说明：如果没有成熟的视频 motion prior，仅靠 SD3 图像先验、少量 TemporalDiTBlock 和单视频 TTA，很难稳定解决帧间漂移。因此下一阶段主线切回 FollowYourCanvas 的 Panda-70M 预训练 checkpoint，把它作为强视频外推 prior，再针对水墨视频做结构化 single-video test-time adaptation。

核心目标不是简单迁移 FollowYourCanvas，而是把任务重新定义为：

> 给定单个水墨视频，利用大规模视频外推 prior、视频内部重复结构和 SAM3 结构语义记忆，实现时序一致的艺术场景补全。

## 原 FollowYourCanvas 的关键事实

- 单次网络窗口固定为 `512 x 512 x 64`。
- 推理整体不是固定尺寸，`target_size` 可以大于单窗口。
- `inference_outpainting-dir.py` 会通过 `get_canvas_size(input_size, target_size, min_overlap, window_size)` 自动计算 outpainting rounds。
- 当目标画布大于单个窗口时，pipeline 使用 `multi_diff_window = [512, 512]` 做 spatial multi-diffusion。
- 原代码使用 Stable Diffusion 2.1 的 2D `AutoencoderKL`，不是 3D VAE。
- 时序稳定主要来自 Panda-70M 训练出的 motion module、64 帧 temporal window、SAM dense embedding、relative_position、multi-diffusion 和 replace/smooth。

原 anchor-target 训练逻辑应保留：

```text
anchor_pixels_values
target_pixels_values
mask
relative_position = [
  target_center_y - anchor_center_y,
  target_center_x - anchor_center_x,
  anchor_h,
  anchor_w,
  target_h,
  target_w
]
```

## 三个核心创新点

### 1. Video-Intrinsic Scene Prior Learning

不要把单个水墨视频当作普通 fine-tuning 数据，而是从视频内部构造大量自监督外推任务，让模型学习该视频自身的构图、风格和运动规律。

实现要点：

- 从目标视频构造多种 anchor-target-overlap 任务。
- 覆盖不同空间位置、外推方向、overlap ratio、尺度和 temporal clip。
- 每个样本保留 full anchor、target、known overlap mask、unknown mask、relative_position、anchor_box、target_box。
- 引入 SinDiffusion-inspired internal sampling：多尺度 resize、patch/grid 覆盖、half-stride overlap、corner/hard-region biased sampling。
- 目标是避免模型只学边缘纹理，增加远区域和四角区域监督。

论文表达：

> We formulate single-video artistic outpainting as video-intrinsic scene prior learning, where multi-scale anchor-target completion tasks are generated from one video.

### 2. SAM3 Structural-Semantic Memory

SAM3 不只是分割器，而是为单视频水墨外推构建两类 memory：

- Structural memory：SAM3 dense image embedding，负责轮廓、边界、局部结构和笔触方向。
- Semantic memory：SAM3 class masks，负责山、水、树、云、人物等未知区域语义规划。

已有目标视频 SAM3 输出：

```text
OUTV-codex-sd3-temporal-outpainting/outputs/sam3_segments/摹阮郜女仙图卷/
  image_embeddings/
  class_masks/
  metadata.json
```

推荐第一版：

- Dataset 读取完整 anchor window 对应的 SAM3 dense embedding。
- 保持 `image_hidden_size: 256`，尽量复用 FYC 原 SAM image condition / image projection / IP cross-attention 分支。
- 读取 SAM3 class masks 训练 semantic layout predictor。
- 推理时先预测 unknown semantic layout，再用 dense structure + predicted layout + relative_position 生成外推区域。

论文表达：

> We introduce SAM3 structural-semantic memory, which decouples local visual structure propagation from long-range semantic layout completion.

### 3. Prior-Preserving Artistic Test-Time Adaptation

不要第一版全量 fine-tune FollowYourCanvas。应保留 Panda-70M 视频 prior，只训练轻量模块适配单个水墨视频。

建议训练：

- SAM3 dense adapter。
- semantic layout predictor。
- image projection / resampler。
- IP cross-attention adapter。
- outpaint condition input layer。
- motion module LoRA 或小型 temporal adapter。

建议冻结：

- VAE。
- text encoder。
- SD2.1 UNet 大部分主干。
- 大部分 motion module 原权重。

建议损失：

```yaml
outpaint_loss_weight: 1.0
semantic_layout_loss_weight: 0.1
boundary_consistency_loss_weight: 0.1
temporal_feature_loss_weight: 0.05
adapter_regularization_weight: 0.01
```

论文表达：

> We personalize a large-scale video outpainting prior through lightweight, prior-preserving adaptation, preserving temporal generation ability while adapting to ink-style structure and texture.

## 实现路线

### Step 1: FollowYourCanvas zero-shot baseline

先确认官方 Panda-70M checkpoint 在目标水墨视频上的真实能力。不加 SAM3，不做 TTA。

本地已发现的默认路径：

```text
pretrained_model_path: /home/610-ltf/DL/models/OUTV/pretrained_models/stable-diffusion-2-1/stable-diffusion-2-1-base
motion_pretrained_model_path: /home/610-ltf/DL/models/OUTV/pretrained_models/follow-your-canvas/checkpoint-40000.ckpt
lmm_path: /home/610-ltf/DL/models/OUTV/pretrained_models/Qwen-VL-Chat
image_pretrained_model_path: /home/610-ltf/DL/models/OUTV/pretrained_models/sam/sam_vit_b_01ec64.pth
target_video: /home/610-ltf/DL/Datasets/data/video/摹阮郜女仙图卷.mp4
```

注意：原 `inference_outpainting-dir.py` 的 `video_dir` 参数要求目录而不是单个 mp4 文件。为了只跑一个视频，建议后续新增安全 wrapper 或准备一个只包含目标视频的输入目录。不要移动原视频。

### Step 2: Single-video intrinsic dataset

把原 Panda-70M dataset 改造成单视频 TTA dataset。每个样本返回：

```text
target_pixels_values
anchor_pixels_values
mask
relative_position
anchor_box
target_box
frame_indices
fps
prompt / empty prompt
```

必须保留完整 anchor window 和 FYC-style overlap，不要退化成简单 center crop inpainting。

### Step 3: SAM3 dense structural memory

读取 `image_embeddings/*.npz`，对齐 frame index 和 anchor_box，接入原 image condition 分支。第一版尽量不重写 adapter。

### Step 4: SAM3 semantic layout memory

读取 `class_masks/*.npz`，训练：

```text
known semantic + outpaint mask + relative_position -> full / unknown semantic layout
```

推理输出：

```text
known_sam3_condition.mp4
predicted_semantic_layout_raw.mp4
predicted_semantic_layout.mp4
```

### Step 5: Prior-preserving TTA

限定 trainable modules，避免破坏视频 prior。每次 checkpoint 后自动推理，输出到训练目录下的 `auto_infer/checkpoint-XXXX/`。

### Step 6: SinDiffusion-inspired curriculum

新增多尺度内部采样、half-stride patch 覆盖和 corner-biased sampling。此方向只迁移 SinDiffusion 的 internal learning 叙事，不迁移模型结构。

### Step 7: Adaptive-size inference

必须保留原 FYC 自适应推理：

```text
input video size -> target_size -> get_canvas_size -> one/multi round -> tile multi-diffusion
```

正式系统不应写死 `384 -> 512`。小尺寸可以作为训练或 sanity，但研究主线需要支持任意 `target_size`。

## 主实验矩阵

| 方法 | FYC Prior | Single-Video TTA | SAM3 Dense | SAM3 Layout | Internal Curriculum |
|---|---|---|---|---|---|
| FYC zero-shot | 是 | 否 | 否 | 否 | 否 |
| FYC full fine-tune | 是 | 是 | 否 | 否 | 否 |
| Ours-A | 是 | 是 | 是 | 否 | 否 |
| Ours-B | 是 | 是 | 是 | 是 | 否 |
| Ours-Full | 是 | 是 | 是 | 是 | 是 |

## 评价维度

- 外推区域结构。
- 水墨风格保持。
- 边界自然度。
- 帧间稳定性。
- 四角/远区域是否有有效内容。
- frame delta / temporal LPIPS。
- boundary error。
- known region consistency。
- semantic layout IoU。
- corner activation。
- unknown 区域低方差比例。

## 执行安全约束

- 不删除、移动、清理 `outputs/`、`checkpoints/`、`datasets/`、`pretrained_models/`。
- 不覆盖原 FollowYourCanvas 配置。
- 新增配置文件，不修改原配置。
- 所有输出放新目录，例如 `outputs/fyc_sam3_tta/`。
- 每次大改前备份关键文件到 `.codex-backup/fyc-sam3-tta-YYYYMMDD-HHMMSS/`。
- 第一阶段只跑 zero-shot baseline；再逐步加入 single-video dataset、SAM3 dense、semantic layout 和 TTA。

