
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ADULT_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "class",
]
GROUPS = {"sex": ("Female", "Male"), "race": ("Non-White", "White")}


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_adult(seed=SEED, device=None, verbose=True, protected="sex", data_dir="data"):
    """Load UCI Adult with the starter's 70/30 split and validation held out from training."""
    device = device or get_device()
    data_dir = Path(data_dir)
    frames = [pd.read_csv(data_dir / filename, names=ADULT_COLUMNS, comment="|",
                          skipinitialspace=True, na_values="?")
              for filename in ("adult.data", "adult.test")]
    df = pd.concat(frames, ignore_index=True).dropna().reset_index(drop=True)
    y = df["class"].str.rstrip(".").eq(">50K").to_numpy(dtype=np.float32)
    z = df[protected].eq(GROUPS[protected][1]).to_numpy(dtype=np.float32)
    features = df.drop(columns=["class", protected, "fnlwgt"])
    pool, test = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed, stratify=y)
    train, val = train_test_split(pool, test_size=0.2, random_state=seed, stratify=y[pool])

    num_cols = features.select_dtypes(include="number").columns.tolist()
    cat_cols = features.columns.difference(num_cols, sort=False).tolist()
    scaler = StandardScaler().fit(features.iloc[train][num_cols])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
    encoder.fit(features.iloc[train][cat_cols])
    names = num_cols + encoder.get_feature_names_out(cat_cols).tolist()
    data = {
        "groups": GROUPS[protected],
        "feature_names": names,
        "n_features": len(names),
        "device": device,
    }
    for split, indices in (("train", train), ("val", val), ("test", test)):
        frame = features.iloc[indices]
        x = np.column_stack([scaler.transform(frame[num_cols]), encoder.transform(frame[cat_cols])])
        data[f"X_{split}"] = torch.tensor(x, dtype=torch.float32, device=device)
        data[f"y_{split}"] = torch.from_numpy(y[indices])
        data[f"z_{split}"] = torch.from_numpy(z[indices])
        if verbose:
            rates = ", ".join(f"{name}: {y[indices][z[indices] == group].mean():.3f}"
                              for group, name in enumerate(data["groups"]))
            print(f"{split}: {len(indices):,} rows, positive label rates ({rates})")
    data["train_ds"] = TensorDataset(data["X_train"], data["y_train"], data["z_train"])
    return data
