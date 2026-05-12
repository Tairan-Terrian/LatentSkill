

<h1 align="center">Mathematical Evolving Memory: Gradient-Free Latent Policy Optimization for Dynamic Agent Skills</h1>




This repository contains the reference code for LatentSkill, including the policy, memory bank, executor, training loop, and evaluation pipelines used across the supported benchmarks.

## 🧩 Overview

LatentSkill represents memory actions with a continuous latent policy and decodes them through a structured operation bank. The implementation in this repository focuses on the full training and evaluation stack, rather than the theory or algorithmic walkthrough.

## ❗ What Is Included

- `main.py`: unified entry point for training and evaluation.
- `src/controller.py`: latent actor, critic, structural decoder, and PPO losses.
- `src/trainer.py`: training orchestration.
- `src/operation_bank.py`: memory operation bank.
- `src/executor.py`: operation execution utilities.
- `src/memory_bank.py`: memory storage and retrieval.
- `src/data_processing/`: dataset-specific preprocessing.
- `src/eval/`: dataset-specific evaluation logic.

## ⭐ Supported Benchmarks

- LoCoMo
- HotpotQA
- LongMemEval-S
- ALFWorld

## 🛠️ Setup
 **
```bash
conda create -n latentskill python=3.10
conda activate latentskill
pip install -r requirements.txt
```


## 🚀 Get Started

Training and evaluation entry points are provided as shell scripts for the main benchmarks:

```bash
bash train_locomo.sh
bash eval_locomo.sh
```

You can also run the unified pipeline directly through `main.py` and adjust the dataset, model, retrieval, and checkpoint arguments as needed.



## 📊 Preparing Training Data

MemSkill builds training and evaluation data from the datasets below. Please download data from the official sources and place them under `data/`. Unless otherwise noted, splits are already configured in our codebase.

### **1) LoCoMo**
- Download LoCoMo from the official repo: [LoCoMo](https://github.com/snap-research/locomo)  
- **Splits**: LoCoMo splits are **already configured in `main.py`** (no extra split file needed).  
- Put the downloaded files under:
  - `data/locomo10.json`

### **2) LongMemEval**
- We use **LongMemEval-S** from: [LongMemEval](https://github.com/xiaowu0162/LongMemEval)  
- **Important**: LongMemEval-S is used for **transfer evaluation only**. That is, skills trained on LoCoMo are **directly evaluated** on LongMemEval-S without additional training.
- Put the downloaded files under:
  - `data/longmemeval_s_cleaned.json`
- Use our split file:
  - `data/longmemeval_s_splits.json` (**We use test split only**)



### **3) HotpotQA**
- Download HotpotQA from: [HotpotQA-Modified](https://huggingface.co/datasets/BytedTsinghua-SIA/hotpotqa/tree/main) (Source: [HotpotQA](https://hotpotqa.github.io/))
- We evaluate on three test files:
  - `data/eval_50.json`
  - `data/eval_100.json`
  - `data/eval_200.json`

These correspond to **increasing context length**, where each query context is constructed by concatenating **50 / 100 / 200 documents** (following the long-context evaluation protocol we adopt in our experiments).



### **4) ALFWorld**
Please follow the official instructions to install dependencies and download assets: [ALFWorld](https://github.com/alfworld/alfworld)

We use **offline expert trajectories** as the interaction corpus for memory construction. We provide a one-command script to collect and save trajectories:

```bash
# Collect expert trajectories for train / seen / unseen splits
python alfworld_replay.py --split train --output ./data/alfworld_train_offline.json
python alfworld_replay.py --split eval_in_distribution --output ./data/alfworld_expert_eval_in_distribution.json
python alfworld_replay.py --split eval_out_of_distribution --output ./data/alfworld_expert_eval_out_of_distribution.json
```

## 📚 Repository Layout

```text
data/        benchmark data files
skills/      task skill templates
src/         core implementation
results/     example outputs and reproductions
checkpoints/ saved checkpoints
```

## 🖥️ Outputs

Typical runs produce checkpoints, result JSON files, cached memory files, and logs under the corresponding output directories.

## 🙏 Notes

- The repository is centered on the LatentSkill implementation and its benchmark integrations.
- Keep API keys and private endpoints out of the README and version control.
