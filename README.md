# Off-Policy Evaluation under Nonignorable Monotone Missingness  

This repository contains the official implementation for the ICML 2025 submission **"Off-Policy Evaluation under Nonignorable Monotone Missingness."** It includes the technical appendix and code for simulations and synthetic data analysis using MIMIC-III, supporting the figures in the paper.  

## Overview  

- **Paper:** "Off-Policy Evaluation under Nonignorable Monotone Missingness"  
- **Conference:** ICML 2025  

## Contents  

To reproduce the results from the paper and appendix:  

1. Set up the environment using `environment.yml`.  
2. Run `scripts/lstdq_sim.py` for value estimation results.  
3. Run `scripts/lstdq_cover.py` to replicate value estimation and inference results (default: \(S=250\) replications).  
4. Run `scripts/data_ope.py` for MIMIC-III data analysis.  
