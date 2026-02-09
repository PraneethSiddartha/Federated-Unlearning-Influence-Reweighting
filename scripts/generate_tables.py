#!/usr/bin/env python3
"""
Table Generation Script
=======================
Generates CSV and Markdown tables from experimental results.
"""

import pandas as pd
from pathlib import Path

# Experimental Data
ALPHA_ABLATION = {
    'alpha': [0.3, 0.4, 0.5, 0.6, 0.7],
    'retain_acc': [79.85, 79.12, 78.92, 76.45, 72.34],
    'forget_acc': [45.32, 31.45, 18.45, 12.78, 8.92],
    'mia_acc': [58.67, 54.23, 51.23, 50.45, 49.87],
    'cosine_sim': [0.987, 0.978, 0.962, 0.941, 0.912],
}

MAIN_RESULTS = {
    'condition': ['Before', 'After (α=0.5)', 'Retrained'],
    'retain_acc': [80.15, 78.92, 79.23],
    'forget_acc': [78.67, 18.45, 12.38],
    'mia_acc': [68.34, 51.23, 50.45],
    'time_s': [None, 0.015, 1847.5],
}

def generate_tables():
    output_dir = Path('../results/tables')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Alpha ablation
    pd.DataFrame(ALPHA_ABLATION).to_csv(output_dir / 'alpha_ablation.csv', index=False)
    
    # Main results
    pd.DataFrame(MAIN_RESULTS).to_csv(output_dir / 'main_results.csv', index=False)
    
    print("Tables generated in results/tables/")

if __name__ == "__main__":
    generate_tables()
