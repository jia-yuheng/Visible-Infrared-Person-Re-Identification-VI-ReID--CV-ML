# AMINet-VI-ReID-CV-ML
Visible-Infrared Person Re-Identification (VI-ReID) with the proposed Adaptive Modality Interaction Network (AMINet). It mitigates modality gap, illumination changes and occlusion via feature learning and cross-modal alignment. It achieves 74.75% Rank-1 on SYSU-MM01, outperforming baseline by 7.93% and state-of-the-art by 3.95%.

## Overview

Visible-Infrared Person Re-Identification (VI-ReID) suffers from severe modality discrepancies between RGB and infrared images, along with challenges such as illumination variation and occlusion.

We propose **AMINet (Adaptive Modality Interaction Network)**, which improves cross-modal feature alignment through:

- **Multi-granularity feature extraction** (full-body + upper-body)
- **Interactive Feature Fusion Strategy (IFFS)** for intra- and cross-modality alignment
- **Phase-Enhanced Structural Attention Module (PESAM)** for illumination-invariant feature learning
- **Adaptive Multi-Scale Kernel MMD (AMK-MMD)** for robust distribution alignment

On SYSU-MM01, AMINet achieves **74.75% Rank-1 accuracy**, outperforming the baseline by **+7.93%** and previous SOTA by **+3.95%**.

## Method

We propose a hierarchical dual-branch framework for cross-modality feature learning:

### 1. HMG-DBNet (Hierarchical Multi-Granular Dual-Branch Network)
- Processes **full-body and upper-body images separately**
- Captures both **global semantic information** and **fine-grained local details**
- Improves robustness to occlusion and background clutter

### 2. IFFS (Interactive Feature Fusion Strategy)
- Combines **intra-modality fusion** with **cross-modality alignment**
- Enables deeper interaction between RGB and IR features
- Produces more discriminative and modality-invariant representations

### 3. PESAM (Phase-Enhanced Structural Attention Module)
- Uses **phase congruency** for illumination-invariant feature extraction
- Applies **edge-guided attention** to focus on key structural regions

### 4. AMK-MMD (Adaptive Multi-Scale Kernel MMD)
- Extends traditional MMD with **multi-scale Gaussian kernels**
- Introduces **adaptive bandwidth and learnable weights**
- Improves alignment across complex feature distributions

## Key Contributions

- A dual-branch multi-granularity framework (HMG-DBNet) for robust VI-ReID
- Interactive Feature Fusion Strategy (IFFS) for joint intra- and cross-modality alignment
- Phase-based structural attention (PESAM) for illumination-invariant representation
- Adaptive multi-scale MMD (AMK-MMD) for flexible and scalable feature alignment
