# A Comparative Study of Transfer Learning Strategies for Low-Resource Sign Language Recognition: From ASL to PSL

A research project investigating which **transfer learning strategy** and **model architecture** best support **Pakistani Sign Language (PSL)** recognition under extreme data scarcity, by transferring knowledge from large American Sign Language (ASL) benchmarks.

**Authors:** Emaim Mohsin, Fatima Amin, Mishaal Mustazhar, Sameeha Shahid
**Institution:** Information Technology University (ITU), Lahore, Pakistan

---

## 1. Overview

Pakistani Sign Language has only **248 public videos across 31 glosses** (the WLPSL dataset), while American Sign Language has large benchmarks such as **WLASL** (2,000+ glosses). This project asks a simple question:

> *Given so little PSL data, which transfer learning strategy and architecture gives the best recognition accuracy?*

To answer this, we built a **three-phase experimental pipeline**:

1. **Phase 1 — ASL Source Validation:** Confirm our model pipelines (I3D and Pose-TGCN) correctly reproduce published results on WLASL-100 *before* touching PSL.
2. **Phase 2 — ASL → FSL Benchmark:** Stress-test the four transfer strategies at scale, using French Sign Language (FSL) as a data-rich proxy target.
3. **Phase 3 — ASL → PSL Benchmark:** Apply the same strategy matrix to the real target — PSL — across three architectures (I3D, TGCN, Pose Transformer).

---

## 2. Key Results

| Experiment | Best Result | Notebook |
|---|---|---|
| ASL I3D source validation (158 videos) | **31.65%** Top-1 | `dl-project-asl__1_.ipynb` |
| ASL → FSL (top-100 glosses) | **59.30%** Top-1 (Full Fine-Tune) | `best-fsl.ipynb` |
| PSL — TGCN (ASL-pretrained) | **26.00%** Top-1 (Partial Fine-Tune) | `dl-psl.ipynb` |
| PSL — Pose Transformer | **30.00%** Top-1 (Full Fine-Tune) — **overall best** | `dl-psl.ipynb` |
| PSL — I3D | 4.00% (checkpoint-loading bug, see §7) | `dl-psl-i3d.ipynb` |

**Headline finding:** Zero-shot accuracy on PSL is ~4% for every architecture — barely above the ~3.2% random baseline — proving that **raw cross-lingual transfer does not work** and that PSL absolutely requires fine-tuning.
---

## 3. Repository Structure

```
.
├── dl-project-asl__1_.ipynb   # Phase 1: ASL source validation (I3D + Pose-TGCN on WLASL-100)
├── best-fsl.ipynb             # Phase 2: ASL → FSL transfer benchmark (Pose Transformer)
├── dl-psl.ipynb               # Phase 3: ASL → PSL benchmark (TGCN + Pose Transformer)
├── dl-psl-i3d.ipynb           # Phase 3: ASL → PSL benchmark (I3D, video-based)
└── Report.docx   # Full written report (methodology, figures, analysis)
```

| Notebook | Phase | Architecture(s) | Role |
|---|---|---|---|
| `dl-project-asl__1_.ipynb` | 1A / 1B | I3D, Pose-TGCN | Reproduce published WLASL-100 accuracy to validate the pipeline |
| `best-fsl.ipynb` | 2 | Pose Transformer | Run all four transfer strategies on a data-rich target (FSL) |
| `dl-psl.ipynb` | 3 | Pose-TGCN, Pose Transformer | Run all four transfer strategies on PSL (pose-based) |
| `dl-psl-i3d.ipynb` | 3 | I3D | Run all four transfer strategies on PSL (raw video) |

---

## 4. Datasets

| Dataset | Role | Scale | Notes |
|---|---|---|---|
| **WLASL-100** | ASL source (video + pose) | 100 glosses | 158–162 of ~200 test videos downloadable (some YouTube links dead) |
| **WLASL poses** | ASL source (FSL pretraining) | 3,607 glosses | Pretrained Pose Transformer reaches 50.72% validation accuracy |
| **LSFB-ISOL (top-100)** | FSL target / validation | 68,262 / 52,477 instances | ~1 GB of pose files; downloaded via `lsfb-dataset` |
| **WLPSL** | PSL target (primary) | 248 videos, 31 glosses | Split: 198 train / 50 test (seed 42) |

---

## 5. Methods

### 5.1 Transfer Learning Strategies

Every architecture is evaluated under the same four strategies:

| Strategy | What is trained | Purpose |
|---|---|---|
| **Zero-shot** | Nothing | Tests raw cross-lingual transfer with no adaptation |
| **Linear probe** | Classification head only | Tests linear separability of frozen source features |
| **Partial fine-tune** | Last layers + head | Minimal adaptation while preserving source knowledge |
| **Full fine-tune** | All weights | Upper bound on achievable accuracy given available data |

### 5.2 Architectures

| Architecture | Input representation | ASL Pretraining Source |
|---|---|---|
| **I3D** | RGB video, 224×224 × 64 frames | `rgb_imagenet.pt` → WLASL-100 fine-tuned checkpoint |
| **Pose-TGCN** | 55-node pose graph | Official WLASL-100 `ckpt.pth` (head re-sized to 31 PSL classes) |
| **Pose Transformer** | 64-frame × 110-dim pose sequence | Pretrained from scratch on WLASL poses for the FSL phase; trained on PSL directly for the PSL phase |

---

## 6. Setup & Reproduction

These notebooks were developed and run on **Kaggle Notebooks** (GPU runtime) and assume Kaggle's `/kaggle/input/` dataset-mounting conventions. To reproduce:

1. Upload/attach the relevant datasets in Kaggle (WLASL videos/poses, LSFB-ISOL, WLPSL) as input datasets.
2. Install per-notebook dependencies (each notebook installs its own requirements in the first cell):

   ```bash
   # dl-project-asl__1_.ipynb
   pip install gdown yt-dlp opencv-python-headless -q

   # best-fsl.ipynb
   pip install lsfb-dataset scikit-learn seaborn pandas -q

   # dl-psl.ipynb
   pip install mediapipe==0.10.14
   ```

3. Run notebooks **in this order** (later phases assume checkpoints/artifacts from earlier ones):
   1. `dl-project-asl__1_.ipynb` — validates the I3D and TGCN pipelines and produces the ASL checkpoints used downstream.
   2. `best-fsl.ipynb` — benchmarks all four strategies on FSL (independent of PSL data).
   3. `dl-psl.ipynb` and `dl-psl-i3d.ipynb` — benchmark all four strategies on PSL using the ASL checkpoints from step 1.

Core stack: **PyTorch**, **OpenCV**, **MediaPipe**, **scikit-learn**, **pandas/NumPy**, **matplotlib/seaborn**.

---
## 7. Future Work
- Complete the ASL Pose-TGCN source validation (Phase 1B) with full Top-1/5/10 metrics.
- Pretrain the Pose Transformer on ASL poses before PSL fine-tuning (currently trained from scratch on PSL).
- Data-fraction ablations (50% / 75% / 100% of PSL training data).
- Attention / saliency visualizations for the Pose Transformer.
- Publish the GitHub repository link in the written report.

