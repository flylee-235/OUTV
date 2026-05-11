**下一步计划**

目标：在 zero-shot baseline 已可接受的基础上，进入单视频 SAM3-TTA 实验线，先做最小 sanity，再逐步加 dense/layout/full 约束，避免一上来长跑浪费时间。

**1. 固化 Zero-Shot Baseline**

记录当前可接受结果：

```bash
outputs/fyc_sam3_tta/zero_shot/20260511-120149/effective_config-2026-05-11-12-01-55/result/摹阮郜女仙图卷.mp4
```

建议保留该 run，不清理。它作为后续 dense/layout/full TTA 的视觉对照基线。

**2. 跑 Dataset + SAM3 读数检查**

先确认训练数据、SAM3 embedding、mask 都能正常读：

```bash
cd /home/610-ltf/DL/models/OUTV/FollowYourCanvas-main

python -B scripts/smoke_test_fyc_sam3_tta.py \
  --video-path /home/610-ltf/DL/Datasets/data/video/摹阮郜女仙图卷.mp4 \
  --sam3-root /home/610-ltf/DL/models/OUTV/OUTV-codex-sd3-temporal-outpainting/outputs/sam3_segments/摹阮郜女仙图卷 \
  --samples 2
```

检查点：
- `anchor_pixels_values`、`target_pixels_values` shape 是否正常；
- `sam3_dense_embeddings` 是否是 `[64, 256, 72, 72]`；
- `sam3_class_masks` 是否和 metadata 类别数量一致；
- mask 值域是否在合理范围内。

**3. Dense-Only TTA Dry-Run**

先只验证配置和输出路径，不真正训练：

```bash
python -B train_fyc_sam3_tta_safe.py \
  --config configs/fyc_sam3_tta_dense.yaml \
  --dry-run
```

重点确认：
- 输出目录在 `outputs/fyc_sam3_tta/dense_only/<timestamp>/`；
- 使用的是单视频数据；
- 使用 precomputed SAM3 dense embedding；
- 没有尝试删除、覆盖旧输出。

**4. Dense-Only 短步数 Sanity**

先跑很短训练，例如 20 到 100 steps。当前配置如果还是 `max_train_steps: 100`，可直接跑：

```bash
torchrun --nnodes=1 --nproc_per_node=1 --master_port=8888 \
  train_fyc_sam3_tta_safe.py \
  --config configs/fyc_sam3_tta_dense.yaml \
  --launcher pytorch
```

检查点：
- loss 是否正常下降或至少稳定；
- 是否能保存 checkpoint；
- 是否没有 OOM；
- 是否没有 shape mismatch；
- 是否能在训练后跑一次推理。

**5. Dense-Only 推理对照**

Dense TTA 训练完成后，用生成的 checkpoint 跑目标视频推理。

对照三组视频：
- zero-shot baseline；
- dense-only TTA；
- 原始输入视频。

重点观察：
- 边界是否更稳定；
- 已知区域是否被破坏；
- 画卷纹理是否更连续；
- 是否出现过拟合导致的局部重复、糊化或闪烁。

**6. Layout-Only TTA**

dense-only 没有明显坏处后，再跑 layout-only：

```bash
python -B train_fyc_sam3_tta_safe.py \
  --config configs/fyc_sam3_tta_layout.yaml \
  --dry-run
```

确认无误后短步数训练。layout-only 主要看语义区域是否更稳定，比如山、水、树、人物边界是否减少漂移。

**7. Full TTA**

dense-only 和 layout-only 都能跑通后，再进入 full：

```bash
python -B train_fyc_sam3_tta_safe.py \
  --config configs/fyc_sam3_tta_full.yaml \
  --dry-run
```

然后短步数 sanity，再决定是否扩大步数。

建议阶段：
- `100 steps`：只看能否跑通；
- `500 steps`：看是否开始改善；
- `1000-2000 steps`：作为第一版正式 TTA；
- 不建议一开始跑很长，单视频很容易过拟合。

**8. 评测与筛选**

每个实验至少保留：

```text
zero_shot
dense_only
layout_only
full_tta
```

每组检查：
- `result/摹阮郜女仙图卷.mp4`
- `original*.mp4`
- `replace*.mp4`
- `smooth*.mp4`
- config 文件
- checkpoint 路径

视觉评测优先级：
1. 已知区域是否保持；
2. outpainting 边界是否自然；
3. 时间连续性是否变好；
4. SAM3 类别布局是否更稳定；
5. 是否出现单视频过拟合伪影。

**推荐你现在执行的下一条命令**

先跑 dataset smoke test：

```bash
cd /home/610-ltf/DL/models/OUTV/FollowYourCanvas-main

python -B scripts/smoke_test_fyc_sam3_tta.py \
  --video-path /home/610-ltf/DL/Datasets/data/video/摹阮郜女仙图卷.mp4 \
  --sam3-root /home/610-ltf/DL/models/OUTV/OUTV-codex-sd3-temporal-outpainting/outputs/sam3_segments/摹阮郜女仙图卷 \
  --samples 2
```

如果通过，就进入 dense-only dry-run。