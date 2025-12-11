# CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval

<p align="center">
  <img src="assets/logo.png" alt="CalibCLIP Logo" width="200"/>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3746027.3755765"><img src="https://img.shields.io/badge/ACM%20MM-2025-blue.svg" alt="ACM MM 2025"></a>
  <a href="https://doi.org/10.1145/3746027.3755765"><img src="https://img.shields.io/badge/DOI-10.1145%2F3746027.3755765-green.svg" alt="DOI"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg" alt="PyTorch"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a>
  <a href="https://python.org/"><img src="https://img.shields.io/badge/Python-3.8%2B-3776ab.svg" alt="Python"></a>
</p>

<p align="center">
  <b>Official PyTorch implementation of CalibCLIP (ACM MM 2025)</b>
</p>

<p align="center">
  <a href="#-highlights">Highlights</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-model-zoo">Model Zoo</a> •
  <a href="#-results">Results</a> •
  <a href="#-citation">Citation</a>
</p>

---

## 📰 News

- **[2025.08]** 🎉 CalibCLIP is accepted by **ACM MM 2025**!
- **[2025.11]** 📦 Code and pretrained models are released.

---

## 🌟 Highlights

<p align="center">
  <img src="assets/framework.png" alt="CalibCLIP Framework" width="90%"/>
</p>

CalibCLIP addresses the **dominant semantics bias** problem in CLIP-based text-driven image retrieval through two key innovations:

### 🔬 Calibrated Vision Encoder (CVE)
- Dynamically recalibrates visual features using **cross-modal contextual information**
- Mitigates over-reliance on dominant visual patterns (e.g., clothing color)
- Enables fine-grained discrimination between visually similar images

### 📚 Dual Curriculum Contrastive Learning (DCC)
- Progressive training strategy with **image-level** and **text-level** curricula
- Gradually increases sample difficulty during training
- Enhances model robustness to hard negative samples
---

## 🛠 Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.7 (recommended)

### Setup Environment

```bash
# Clone repository
git clone https://github.com/your-username/CalibCLIP.git
cd CalibCLIP

# Create conda environment
conda create -n calibclip python=3.9 -y
conda activate calibclip

# Install PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Install CalibCLIP
pip install -e .
```

### requirements.txt

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
scipy>=1.7.0
Pillow>=9.0.0
PyYAML>=6.0
tqdm>=4.62.0
tensorboard>=2.10.0
matplotlib>=3.5.0
ftfy>=6.1.0
regex>=2022.1.18
```

---

## 🚀 Quick Start

### Single Model Evaluation

```bash
python scripts/evaluate.py \
    --config configs/cuhkpedes.yaml \
    --checkpoint checkpoints/calibclip_cuhkpedes.pth \
    --data_root /path/to/CUHK-PEDES \
    --output_dir results/
```

### Batch Evaluation (Multiple Datasets)

```bash
python scripts/batch_evaluate.py \
    --checkpoint checkpoints/calibclip_cuhkpedes.pth \
    --datasets cuhkpedes icfgpedes rstp \
    --data_roots /path/to/CUHK-PEDES /path/to/ICFG-PEDES /path/to/RSTPReid \
    --output_dir results/batch/
```

### Interactive Demo

```bash
python scripts/demo.py \
    --config configs/cuhkpedes.yaml \
    --checkpoint checkpoints/calibclip_cuhkpedes.pth \
    --image_dir /path/to/images \
    --query "A woman wearing a red dress"
```

### Python API

```python
from calibclip import CalibCLIP, build_calibclip
from calibclip.datasets import build_transforms
from calibclip.evaluation import Evaluator

# Build model
model = build_calibclip(
    pretrained="checkpoints/calibclip_cuhkpedes.pth",
    embed_dim=512,
    vision_layers=12,
    text_layers=12,
)
model.eval()

# Build transforms
transform = build_transforms(is_train=False, image_size=(384, 128))

# Run evaluation
evaluator = Evaluator(model, device="cuda")
results = evaluator.evaluate(dataloader)
print(f"Rank-1: {results['rank1']:.2f}%")
```

---

## 📊 Datasets

### Supported Datasets

| Dataset | Images | IDs | Captions | Download |
|---------|--------|-----|----------|----------|
| [CUHK-PEDES](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) | 40,206 | 13,003 | 80,412 | [Link](https://github.com/ShuangLI59/Person-Search-with-Natural-Language-Description) |
| [ICFG-PEDES](https://github.com/zifyloo/SSAN) | 54,522 | 4,102 | 54,522 | [Link](https://github.com/zifyloo/SSAN) |
| [RSTPReid](https://github.com/NjtechCVLab/RSTPReid-Dataset) | 20,505 | 4,101 | 41,010 | [Link](https://github.com/NjtechCVLab/RSTPReid-Dataset) |

### Dataset Preparation

```
data/
├── CUHK-PEDES/
│   ├── imgs/
│   │   ├── cam_a/
│   │   ├── cam_b/
│   │   └── ...
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── ICFG-PEDES/
│   ├── imgs/
│   └── annotations/
└── RSTPReid/
    ├── imgs/
    └── annotations/
```

---

## 🏆 Model Zoo

### Pretrained Models

| Model | Dataset | Rank-1 | Rank-5 | Rank-10 | mAP | 
|-------|---------|--------|--------|---------|-----|
| CalibCLIP-B/16 | CUHK-PEDES | **78.35** | 90.12 | 93.87 | 69.42 | 
| CalibCLIP-B/16 | ICFG-PEDES | **65.28** | 81.45 | 86.92 | 42.16 | 
| CalibCLIP-B/16 | RSTPReid | **62.15** | 82.30 | 89.45 | 48.73 | 

> 📌 All models use ViT-B/16 as the vision backbone, initialized from OpenAI CLIP.

---

## 📈 Results

### Comparison with State-of-the-Art

We conduct comprehensive experiments on three text-based person retrieval benchmarks.

#### Methods without CLIP Backbone

| Method | Venue | CUHK-PEDES ||| ICFG-PEDES ||| RSTPReid |||
|--------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |
| EAIBC | TNNLS'24 | 64.96 | 83.36 | 88.42 | 58.95 | 75.95 | 81.72 | 49.85 | 70.15 | 79.85 |
| IVT | ECCV'22 | 65.59 | 83.11 | 89.20 | 56.04 | 73.60 | 80.22 | 46.70 | 70.00 | 78.80 |
| CTLG | TCSVT'23 | 69.47 | 87.13 | 92.13 | 57.69 | 75.79 | 82.67 | - | - | - |
| SAP-SAM | MM'24 | 75.05 | 89.93 | 93.73 | 63.97 | 80.84 | 86.17 | 62.85 | 82.65 | 89.85 |

#### Methods with CLIP Backbone

| Method | Venue | CUHK-PEDES |||| ICFG-PEDES |||| RSTPReid ||||
|--------|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| | | R@1 | R@5 | R@10 | mAP | R@1 | R@5 | R@10 | mAP | R@1 | R@5 | R@10 | mAP |
| CFine | TIP'23 | 69.57 | 85.93 | 91.15 | - | 60.83 | 76.55 | 82.42 | - | 50.55 | 72.50 | 81.60 | - |
| IRRA | CVPR'23 | 73.38 | 89.93 | 93.71 | 66.13 | 63.46 | 80.25 | 85.82 | 38.06 | 60.20 | 81.30 | 88.20 | 47.17 |
| TILT | MM'24 | 74.46 | 90.21 | 94.19 | 66.31 | 63.77 | 80.80 | 86.00 | 38.07 | 60.75 | 81.80 | 88.70 | 47.56 |
| IRLT | AAAI'24 | 74.46 | 90.19 | 94.01 | - | 64.72 | 81.35 | 86.31 | - | 61.49 | 82.26 | 89.23 | - |

### 🔌 CalibCLIP as Plug-and-Play Module

CalibCLIP can be seamlessly integrated into existing CLIP-based methods as a plug-and-play module:

#### On CLIP-ViT/16 Baseline

| Dataset | Metric | Baseline | +CalibCLIP | Improvement |
|---------|--------|:--------:|:----------:|:-----------:|
| **CUHK-PEDES** | R@1 | 66.54 | **71.88** | +5.34 |
| | R@5 | 86.94 | **90.50** | +3.56 |
| | R@10 | 91.77 | **94.75** | +2.98 |
| | mAP | 62.69 | **65.22** | +2.53 |
| **ICFG-PEDES** | R@1 | 57.44 | **62.54** | +5.10 |
| | R@5 | 75.79 | **80.18** | +4.39 |
| | R@10 | 82.22 | **84.57** | +2.35 |
| | mAP | 33.03 | **37.37** | +4.34 |
| **RSTPReid** | R@1 | 56.67 | **60.30** | +3.63 |
| | R@5 | 78.09 | **82.78** | +4.69 |
| | R@10 | 86.62 | **88.66** | +2.04 |
| | mAP | 44.25 | **46.47** | +2.22 |



## 🔧 Configuration

### Base Configuration

```yaml
# configs/base.yaml
model:
  name: "calibclip"
  embed_dim: 512
  vision_layers: 12
  vision_width: 768
  vision_patch_size: 16
  text_layers: 12
  text_width: 512
  text_heads: 8
  vocab_size: 49408

  # CVE settings
  cve:
    enabled: true
    num_heads: 8
    num_layers: 2
    dropout: 0.1
  
  # DCC settings
  dcc:
    enabled: true
    temperature: 0.07
    image_curriculum_epochs: [10, 20, 30]
    text_curriculum_epochs: [15, 25, 35]

data:
  image_size: [384, 128]
  max_length: 77

evaluation:
  batch_size: 128
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_models.py -v

# Test with coverage
pytest tests/ --cov=calibclip --cov-report=html
```

---

## 📝 Training (Coming Soon)

Training code will be released soon. Stay tuned!

```bash
# Training example (placeholder)
python tools/train.py \
    --config configs/cuhkpedes.yaml \
    --data_root /path/to/CUHK-PEDES \
    --output_dir outputs/
```

---

## 🙏 Acknowledgements

This project is built upon the following excellent works:

- [CLIP](https://github.com/openai/CLIP) - Contrastive Language-Image Pre-training
- [IRRA](https://github.com/anosorae/IRRA) - Cross-Modal Implicit Relation Reasoning
- [TransReID](https://github.com/damo-cv/TransReID) - Transformer-based Object Re-Identification

We thank the authors for their valuable contributions to the community.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

For questions or discussions, please:
- Open an issue in this repository
- Contact: [your-email@example.com](mailto:your-email@example.com)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/CalibCLIP&type=Date)](https://star-history.com/#your-username/CalibCLIP&Date)

---

## 📖 Citation

If you find CalibCLIP useful in your research, please consider citing:

```bibtex
@inproceedings{10.1145/3746027.3755765,
  author = {Kang, Bin and Chen, Bin and Wang, Junjie and Li, Yulin and Zhao, Junzhi and Wang, Junle and Tian, Zhuotao},
  title = {CalibCLIP: Contextual Calibration of Dominant Semantics for Text-Driven Image Retrieval},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3755765},
  doi = {10.1145/3746027.3755765},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {5140–5149},
  numpages = {10},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```

---

<p align="center">
  Made with ❤️ by the CalibCLIP Team
</p>