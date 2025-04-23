# imports
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# configuration
CSV_PATH = Path(
    "/Users/kajaloennecken/Documents/Supernovae-ML/Supernovae/"
    "filtered_specific_type_II_supernovae_with_label.csv"
)
OUT_PATH = CSV_PATH.with_name(CSV_PATH.stem + "_augmented.csv")

KEEP_COLS = ["Gi", "GVhel", "mag", "Type", "Type II Label"]
NUMERIC = ["Gi", "GVhel", "mag"]
TARGET = "Gi"
PREDICTORS = ["GVhel", "mag"]
TARGET_TOTAL = 1_000  # desired final size
SEED = 42

# load and basic cleaning
df = pd.read_csv(CSV_PATH, usecols=lambda c: c in KEEP_COLS)

for col in NUMERIC:  # force numeric → NaNs on bad parses
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=KEEP_COLS).reset_index(drop=True)
orig_len = len(df)
print(f"Rows after cleaning : {orig_len}")

#  decide how many synthetic rows are needed
if orig_len >= TARGET_TOTAL:
    print("Dataset already has ≥ 1 000 clean rows; no augmentation needed.")
    df[KEEP_COLS].to_csv(OUT_PATH, index=False)
    print(f"Cleaned file saved : {OUT_PATH}")
    raise SystemExit

N_SYN = TARGET_TOTAL - orig_len
print(f"Synthetic rows to generate: {N_SYN}")

# fit regression & residual scatter
X = df[PREDICTORS].values
y = df[TARGET].values
reg = LinearRegression().fit(X, y)
sigma = (y - reg.predict(X)).std(ddof=1)

rng = np.random.default_rng(SEED)

# generate synthetic rows
synthetic_rows = []
for _ in range(N_SYN):
    base = df.sample(1, random_state=rng).iloc[0]
    noise = rng.normal(0, sigma)
    gi_new = float(reg.predict(base[PREDICTORS].values.reshape(1, -1)) + noise)

    new_row = base.copy()
    new_row[TARGET] = gi_new
    synthetic_rows.append(new_row)

df_syn = pd.DataFrame(synthetic_rows)

# merge and save
df_aug = pd.concat([df, df_syn], ignore_index=True)[KEEP_COLS]
assert len(df_aug) == TARGET_TOTAL, "Final dataset size mismatch!"

df_aug.to_csv(OUT_PATH, index=False)

print(f"Final rows (clean + synthetic): {len(df_aug)}")
print(f"Augmented CSV written to      : {OUT_PATH}")
