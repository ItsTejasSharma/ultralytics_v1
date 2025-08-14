# A Dual Track YOLO-based Network for fire and smoke detection

## Overview

This repository contains the implementation of our improved YOLO-based approach for fire and smoke detection. Our method achieves state-of-the-art detection accuracy on the D-Fire dataset, offering enhanced precision in challenging scenarios. 

We propose a novel deep learning model that, for the first time in this domain, integrates a dual-branch structure combining CNN and Swin Transformer. The model enhances multi-scale feature representation through SPPF, optimizes channel information via ECA, and achieves efficient feature fusion with BiFPN. Furthermore, the detection head employs a decoupled structure to improve overall performance and robustness.

## Base Framework

- **Base Model**: YOLO11 (Ultralytics implementation)
- **Original Repository**: [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## Key Modifications and Contributions

### 1. Configuration Files (YAML)
- **Location**: `Ultralytics/ultralytics/cfg/models/yamls/` directory
- **Purpose**: Custom training configurations, model architectures, and hyperparameter settings
- **Key Files**:
  - `CNN_track.yaml` - Architecture of CNN-Track composed of Depthwise Separable C3k2 (DWC3k2) blocks. The CNN track specializes in local feature extraction.
  - `Swin_track.yaml` - Architecture of Swin-Track consisting of Swin transformer blocks. The Swin track specializes in global feature extraction.
  - `dualTrack.yaml` - Architecture of the Dual track configuration concatenating feature maps at three distinct scales.
  - `dualTrack_attention.yaml` - Architecture of Dual track configuration with Efficient Channel Attention (ECA) module , which enhances channel-wise feature representations with minimal computational overhead.
  - `proposedNetwork.yaml` - Architecture of the proposed network encompassing the dual-branch structure, ECA and Bidirectional Feature Pyramid Network (BiFPN).


### 2. Custom Modules
- **Location**: `custom_modules/` directory
- **Description**: Modified and new neural network components
- **Key Components**:
  - Custom backbone architectures
  - Modified neck/head structures
  - Novel attention mechanisms
  - Custom loss functions

### 3. Custom Functions
- **Location**: `utils/custom_utils.py` and various module files
- **Functionality**:
  - Data preprocessing pipelines
  - Custom augmentation techniques
  - Evaluation metrics
  - Visualization tools

### 4. Training Pipeline Modifications
- Enhanced data loading mechanisms
- Custom callback functions
- Modified optimizer configurations
- Advanced learning rate scheduling

## Installation

### Prerequisites
```bash
pip install ultralytics
pip install torch torchvision
pip install opencv-python
pip install numpy pandas matplotlib
# Add any other specific dependencies
```

### Setup
1. Clone this repository:
```bash
git clone [your-repo-url]
cd [your-repo-name]
```

2. Install additional custom dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train_custom.py --config configs/custom_model.yaml --data configs/dataset_config.yaml
```

### Inference
```bash
python detect_custom.py --weights path/to/custom_weights.pt --source path/to/images
```

### Evaluation
```bash
python evaluate_custom.py --weights path/to/custom_weights.pt --data configs/dataset_config.yaml
```

## File Structure

```
├── configs/
│   ├── custom_model.yaml
│   ├── training_config.yaml
│   └── dataset_config.yaml
├── custom_modules/
│   ├── __init__.py
│   ├── custom_backbone.py
│   ├── custom_head.py
│   └── custom_losses.py
├── utils/
│   ├── custom_utils.py
│   ├── data_processing.py
│   └── visualization.py
├── scripts/
│   ├── train_custom.py
│   ├── detect_custom.py
│   └── evaluate_custom.py
├── requirements.txt
└── README.md
```

## Key Features and Improvements

1. **[Feature 1]**: Brief description of what it does and why it's beneficial
2. **[Feature 2]**: Brief description of what it does and why it's beneficial
3. **[Feature 3]**: Brief description of what it does and why it's beneficial

## Experimental Results

### Performance Metrics
- **mAP@0.5**: [Your results]
- **mAP@0.5:0.95**: [Your results]
- **Inference Speed**: [Your results]
- **Model Size**: [Your results]

### Comparison with Base Model
| Metric | Base YOLO | Custom Model | Improvement |
|--------|-----------|--------------|-------------|
| mAP@0.5 | [value] | [value] | +[%] |
| mAP@0.5:0.95 | [value] | [value] | +[%] |
| Speed (FPS) | [value] | [value] | +[%] |

## Dataset

- **Dataset Name**: [Your dataset]
- **Number of Classes**: [Number]
- **Training Images**: [Number]
- **Validation Images**: [Number]
- **Test Images**: [Number]

## Important Notes for Users

⚠️ **Compatibility Warning**: This customized version may not be directly compatible with standard Ultralytics YOLO workflows due to architectural modifications.

📝 **Configuration**: Always use the provided YAML configuration files when training or running inference to ensure proper model behavior.

🔧 **Dependencies**: Some custom modules may require additional dependencies not present in the base Ultralytics package.

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

## Contributing

This is a research-focused project. For questions or collaboration inquiries, please contact [your-email@domain.com].

## License

[Specify your license - typically should be compatible with Ultralytics' license]

## Acknowledgments

- Original YOLO implementation by Ultralytics
- [Any other acknowledgments for datasets, collaborators, etc.]

---

**Note**: This README provides an overview of the customizations made to the base YOLO model. For detailed implementation specifics, please refer to the respective source files and configuration documents.
