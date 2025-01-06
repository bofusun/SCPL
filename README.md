# Salience-Invariant Consistent Policy Learning for Generalization in Visual Reinforcement Learning

Code for "Salience-Invariant Consistent Policy Learning for Generalization in Visual Reinforcement Learning", AAMAS 2025 paper.

---
# Introduction

we propose the Salience-Invariant Consistent Policy Learning(SCPL) algorithm, an efficient framework for zero-shot generalization in visual reinforcement learning. SCPL utilizes a novel value consistency module to encourage the encoder and value function to capture task-relevant pixels in original and perturbed observations. Meanwhile, a dynamics module is proposed to generate dynamic and reward relevant representations for both observations. Furthermore, SCPL regularizes the policy network using a KL divergence constraint between the policies for original and augmented observations, enabling agents to make consistent decisions in test environments. Experimental results demonstrate SCPL's superior performance over state-of-the-art baselines.
![1736156926168](https://github.com/user-attachments/assets/ceb7ff91-62e9-4374-b61e-024370376fee)

---
# Quick Start

1. Setting up repo
```
git clone https://github.com/bofusun/SCPL
```
2. Install Dependencies
```
conda create -n SCPL python=3.8
conda activate SCPL
cd SCPL
pip install -r requirements.txt
```
3. Train
   
(1) train scpl without dynamic module with random convolution augmentation
```
python src/my_train_all.py --domain_name walker --task_name walk --algorithm scpl0r --seed 0
```
(2) train scpl without dynamic module with overlay augmentation
```
python src/my_train_all.py --domain_name walker --task_name walk --algorithm scpl0 --seed 0
```
(3) train scpl with random convolution augmentation
```
python src/my_train_all.py --domain_name walker --task_name walk --algorithm scplr --seed 0
```
(4) train scpl without dynamic module with overlay augmentation
```
python src/my_train_all.py --domain_name walker --task_name walk --algorithm scpl --seed 0
```
