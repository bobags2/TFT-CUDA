#!/usr/bin/env python3
"""Check CSV data files structure."""

import pandas as pd
from pathlib import Path

data_dir = Path("data/")
csv_files = list(data_dir.glob("*10m*.csv"))

for csv_file in csv_files:
    print(f"\n{csv_file.name}:")
    df = pd.read_csv(csv_file, nrows=5)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape: {pd.read_csv(csv_file).shape}")
    print(f"  First row sample:")
    print(f"    Date: {df.iloc[0, 0] if len(df) > 0 else 'N/A'}")
    print(f"    Time: {df.iloc[0, 1] if len(df) > 0 and len(df.columns) > 1 else 'N/A'}")
