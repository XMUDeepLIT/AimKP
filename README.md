# AimKP: Augmenting Intra-Modal Understanding in MLLMs for Robust Multimodal Keyphrase Generation (AAAI-2026)

## Install
1. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install -r requirements.txt
```
2. Training
Image data of CMKP dataset can be found in [CMKP repo](https://github.com/yuewang-cuhk/CMKP)
MLLM: LLaVA-v1.5-7b, Vision Encoder: clip-vit-large-patch14-336
```Shell
# For standard training
bash /path/to/standard_finetune.sh
# For training under AimKP
bash /path/to/scripts/AimKP.sh
```
3. Evaluation
```Shell
python evaluate.py --model-path checkpoint --model-base /path/to/models/llava-v1.5-7b --txt-path "reults"
```
## Acknowledgement
- Code is based on [LLaVA](https://github.com/haotian-liu/LLaVA)