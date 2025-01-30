# Off-Policy Evaluation under Nonignorable Monotone Missingness  

This repository contains the official implementation for the ICML 2025 submission **"Off-Policy Evaluation under Nonignorable Monotone Missingness."** It includes the technical appendix and code for simulations and synthetic data analysis using MIMIC-III, supporting the figures in the paper.  

## Overview  

- **Paper:** "Off-Policy Evaluation under Nonignorable Monotone Missingness"  
- **Conference:** ICML 2025  
  
## Repository Structure  

- **`scripts/`** – Contains the main scripts for simulations and MIMIC-III data analysis.
- **`batch_rl/`** – Includes code for learning the optimal policy from other reinforcement learning (RL) approaches. These policies are used as target policies in the MIMIC-III data analysis of the main paper.  
- **`custom_env/`** – Stores the environments used in both simulation and MIMIC-III data analysis.  
- **`ope_mnar/`** – Includes core Python files that support the execution of the scripts.  

## Reproducing Results  

To replicate the results presented in the paper and appendix, follow these steps:  
 

1. Set up the environment using `environment.yml`.
2. Create a subfolder `ope_mnar/output` to save results.
3. Run `scripts/lstdq_sim.py` for value estimation results.  
4. Run `scripts/lstdq_cover.py` to replicate value estimation and inference results (default: $S=250$ replications).  
5. Run `scripts/data_ope.py` for MIMIC-III data analysis.  
