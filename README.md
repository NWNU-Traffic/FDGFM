# Structural Feature Fusion in Frequency and Directional Domains for Robust Traffic Sign Detection

This repository provides the source code for the manuscript:

**Structural Feature Fusion in Frequency and Directional Domains for Robust Traffic Sign Detection**
Submitted to *The Visual Computer*.

The repository contains the RT-DETR implementation used in the manuscript and the proposed **Frequency-Domain Directional Gated Fusion Module (FDGFM)**.

FDGFM contains three main components:

* **Patch-level Spectral Enhancement (PSE)**
* **Omnidirectional Depthwise Convolution (ODC)**
* **Adaptive Branch Fusion (ABF)**

FDGFM is designed as a plug-and-play structural feature enhancement module for traffic sign detection. It complements standard spatial features with local frequency-domain responses, directional context, and adaptive branch fusion.

## Overview

Traffic signs in complex road scenes often contain fragile structural cues, such as weak boundaries, digits, arrows, text, and geometric contours. These cues can be degraded by long-distance imaging, illumination changes, adverse weather, occlusion, motion blur, and background textures.

The proposed FDGFM addresses this problem by combining:

1. **Local frequency-domain recalibration** for boundary, symbol, and texture details.
2. **Directional context modeling** for continuous sign edges, poles, arrows, and elongated contours.
3. **Spatially adaptive branch fusion** for integrating semantic, frequency-enhanced, and directional responses.

The module preserves the input and output feature dimensions, so it can be inserted into detector feature layers without modifying the detection head.

## Repository Structure

```text
FDGFM/
└── RT-DETR/
    └── ultralytics/
        ├── cfg/
        │   └── models/
        │       └── rt-detr/
        │           └── rtdetr-fdgfm.yaml     # RT-DETR configuration with FDGFM
        └── nn/
            └── Changemodules/
                └── fdgfm.py                 # Core implementation of PSE, ODC, ABF, and FDGFM
```

The FDGFM-related implementation mainly includes two files.

* `RT-DETR/ultralytics/nn/Changemodules/fdgfm.py` provides the core implementation of PSE, ODC, ABF, and the complete FDGFM module.
* `RT-DETR/ultralytics/cfg/models/rt-detr/rtdetr-fdgfm.yaml` provides the RT-DETR configuration used to integrate FDGFM into the detector.

Other files follow the RT-DETR/Ultralytics-style project structure and are used for training, validation, inference, visualization, and model evaluation.

## Installation

Clone this repository:

```bash
git clone https://github.com/NWNU-Traffic/FDGFM.git
cd FDGFM/RT-DETR
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The implementation is based on PyTorch and follows the RT-DETR/Ultralytics-style code structure.

## Core Module

The main implementation file is:

```text
RT-DETR/ultralytics/nn/Changemodules/fdgfm.py
```

This file includes:

* `PatchSpectralEnhancer`: patch-level spectral enhancement module.
* `OrientedDepthwiseContext`: omnidirectional depthwise context module.
* `AdaptiveBranchFusion`: adaptive fusion module.
* `FDGFM`: complete frequency-domain directional gated fusion module.

The configuration file used for RT-DETR integration is:

```text
RT-DETR/ultralytics/cfg/models/rt-detr/rtdetr-fdgfm.yaml
```

## Datasets and Sources

The experiments in the manuscript use publicly available datasets from their original providers or public dataset pages. This repository does not redistribute these datasets. Please download each dataset from the corresponding source and follow its license and usage terms.

| Dataset     | Source                                                | URL                                                                                     |
| ----------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| CCTSDB2021  | Official GitHub repository                            | https://github.com/csust7zhangjm/CCTSDB2021                                             |
| TT100K      | Official Tsinghua-Tencent 100K dataset page           | https://cg.cs.tsinghua.edu.cn/traffic-sign/                                             |
| GTSDB       | Official German Traffic Sign Detection Benchmark page | https://benchmark.ini.rub.de/gtsdb_dataset.html                                         |
| RoadSign    | Kaggle Road Sign Detection dataset page               | https://www.kaggle.com/datasets/andrewmvd/road-sign-detection                           |
| VNTS        | Kaggle Vietnam Traffic Signs dataset page             | https://www.kaggle.com/datasets/maitam/vietnamese-traffic-signs                         |
| COCO subset | Kaggle COCO 25-Class Object Detection dataset page    | https://www.kaggle.com/datasets/malaychand/coco-25-class-object-detection-yolo-datasets |

The COCO subset was used only as supplementary validation to examine the transferability boundary of FDGFM beyond traffic sign datasets.

Users should cite the original dataset papers or dataset pages when using these datasets.

## Training and Evaluation

The project follows the RT-DETR/Ultralytics-style training and validation workflow.

Before training, please prepare the dataset configuration files and check the paths in the corresponding YAML files.

A typical training command can be organized as follows:

```bash
python train.py
```

A typical validation command can be organized as follows:

```bash
python val.py
```

Please adjust the dataset paths, model configuration, image size, batch size, and other settings according to your local environment and experimental protocol.

## Visualization and Analysis

The repository also contains scripts for model evaluation and visualization, including feature response visualization and inference-speed measurement.

Examples include:

```text
heatmap.py
get_FPS.py
get_COCO_metrice.py
get_all_yaml_param_and_flops.py
```

These scripts are used for visual analysis, speed evaluation, metric calculation, and model-complexity analysis.

## Code Availability Statement

The source code is available at:

```text
https://github.com/NWNU-Traffic/FDGFM
```

## Citation

If you use this code or the implemented module, please cite the corresponding manuscript:

```bibtex
@article{FDGFM2026,
  title={Structural Feature Fusion in Frequency and Directional Domains for Robust Traffic Sign Detection},
  author={Ma, X. and Ma, D. and He, X.},
  journal={The Visual Computer},
  note={Submitted},
  year={2026}
}
```

Please update the citation after the final publication information becomes available.

## License and Usage

This repository is released for academic research use. Please follow the licenses of the original RT-DETR/Ultralytics-style codebase and the datasets used in the manuscript.

The datasets are not redistributed in this repository.

## Contact

For questions about this repository, please contact the authors or open an issue in the GitHub repository.
