# A Dual Track YOLO-based Network for fire and smoke detection

## Overview

This repository contains the implementation of our improved YOLO-based approach for fire and smoke detection. Our method achieves state-of-the-art detection accuracy on the D-Fire dataset, offering enhanced precision in challenging scenarios. 

We propose a novel deep learning model that, for the first time in this domain, integrates a dual-branch structure combining CNN and Swin Transformer. The model enhances multi-scale feature representation through SPPF, optimizes channel information via ECA, and achieves efficient feature fusion with BiFPN. Furthermore, the detection head employs a decoupled structure to improve overall performance and robustness.

## Key Modifications and Contributions

### 1. Configuration Files (YAML)
- **Location**: `Ultralytics/ultralytics/cfg/models/yamls/` directory
- **Purpose**: Custom training configurations, model architectures, and hyperparameter settings
- **Key Files**:
  - `CNN_track.yaml` - Architecture of CNN-Track composed of Depthwise Separable C3k2 (DWC3k2) blocks.
  - `Swin_track.yaml` - Architecture of Swin-Track consisting of Swin transformer blocks. 
  - `dualTrack.yaml` - Architecture of the Dual track configuration concatenating feature maps at three distinct scales.
  - `dualTrack_attention.yaml` - Architecture of Dual track configuration with Efficient Channel Attention (ECA) module , which enhances channel-wise feature representations with minimal computational overhead.
  - `proposedNetwork.yaml` - Architecture of the proposed network encompassing the dual-branch structure, ECA and Bidirectional Feature Pyramid Network (BiFPN).
    
## Installation

### Prerequisites
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install numpy pandas matplotlib
```

### Setup
1. Clone this repository:
```bash
!git clone https://github.com/ItsTejasSharma/ultralytics.git
%cd ultralytics
!pip install -e .
from ultralytics import YOLO
```

## Usage

### Training
```bash
!yolo train model=proposedModel.yaml data=data_yaml.yaml epochs=100 imgsz=640 device="cpu"
```

### Perform object detection on an image
```bash
results = model("path/to/image.jpg")  # Predict on an image
results[0].show()  # Display results
```

### Evaluate the model's performance on the validation set
```bash
metrics = model.val()
```

## Key Features and Improvements

1. **Dual-Track Backbone**: It combines a CNN track that extracts low-level features and a Swin transformer track that extracts global features. This dual-track design synergistically leverages the complementary strengths of CNNs and Transformers to enhance the accuracy and robustness of fire and smoke detection.
2. **Depthwise Separable C3k2 (DWC3k2) Blocks**: DWC3k2 Refines feature maps through depthwise-separable convolutions.It enhances efficiency while maintaining strong feature extraction capabilities compared to standard C3k2
3. **BiFPN & ECA**: Bidirectional Feature Pyramid Network (BiFPN) is integrated into the neck of the proposed dual-track network to enhance multi-scale feature fusion. Additionally, Efficient Channel Attention (ECA) mechanism is included which sharpens the focus on subtle fire and smoke cues, boosting feature quality for higher accuracy.

## Experimental Results

### Performance Metrics
- **mAP@0.5**: 
- **Precison**: [Your results]
- **Recall**: [Your results]
- **GFLOPs**: [Your results]
- **FPS**:
- **Inference Time**:

### Comparison with State-of-the-art methods for object detection on the DFire dataset.
| Model    | Precision (%) | Recall (%) | mAP@0.5 (%) | GFLOPs | FPS   | Inference Time (ms) |
|----------|---------------|------------|-------------|--------|-------|---------------------|
| YOLOv5m  | 75.3          | 67.0       | 74.6        | 64.4   | 43.99| 22.73           |
| YOLOv8m  | 76.8          | 70.6  | 77.7    | 79.1   | 37.98 | 26.32               |
| YOLOv9m  | **77.3**      | 69.4       | 76.9        | 77.6   | 37.18 | 26.89               |
| YOLOv10m | 75.4          | 68.7       | 75.5        | 64.0 | 42.96 | 23.27               |
| YOLOv11m | 75.3          | 68.4       | 75.2        | 68.2   | 40.52 | 24.68               |
| YOLOv12m | 76.0          | 67.8       | 74.9        | 68.1   | 33.47 | 29.87               |
| RT-DETR  | 76.2          | 69.9       | 75.4        | 108.0 | 18.84 | 53.07               |
| Proposed Network  | **79.4**          | **75.7**       | **81.0**        | **58.2** | **20.67** | **48.38**               |


## Dataset

D-Fire is an image dataset of fire and smoke occurrences designed for machine learning and object detection algorithms with more than 21,000 images. The dataset summary is detailed in the table below.

<div align="center">
<table>
  <tr>
    <th>Number of images</th>
    <th>Number of bounding boxes</th>
  </tr>
 
  <tr><td>

  | Category | # Images |
  | ------------- | ------------- |
  | Only fire  | 1,164  |
  | Only smoke  | 5,867  |
  | Fire and smoke  | 4,658  |
  | None  | 9,838  |

  </td><td>

  | Class | # Bounding boxes |
  | ------------- | ------------- |
  | Fire  | 14,692 |
  | Smoke  | 11,865 |

  </td></tr> 
</table>
</div>

Annotations follow the YOLO format, with normalized coordinates between 0 and 1.

## Citation

If you use this work in your research, please cite:

```bibtex
@article{your_paper,
    title={Your Paper Title},
    author={Your Name and Co-authors},
    journal={Journal/Conference Name},
    year={Year},
    volume={Volume},
    pages={Pages}
}
```

## Acknowledgments

-  **Original YOLO implementation by Ultralytics**: [ultralytics](https://github.com/ultralytics/ultralytics)

- D-Fire Dataset:
  Pedro Vinícius Almeida Borges de Venâncio, Adriano Chaves Lisboa, Adriano Vilela Barbosa: An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained    devices. In: Neural Computing and Applications, 2022.



---

**Note**: This README provides an overview of the customizations made to the base YOLO model. For detailed implementation specifics, please refer to the respective source files and configuration documents.
