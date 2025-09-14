import os
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def _mm_root(root: str, dataset: str) -> Path:
	return Path(root) / dataset / "memmap"


def available(root: str, dataset: str) -> bool:
	return (_mm_root(root, dataset) / "READY").exists()


def save_memmap(root: str, dataset: str, x, y, edge_index, splits: dict) -> None:
	mm_dir = _mm_root(root, dataset)
	mm_dir.mkdir(parents=True, exist_ok=True)

	# ensure numpy arrays on CPU
	x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
	y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
	ei_np = edge_index.detach().cpu().numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)

	meta = {
		"X_shape": x_np.shape, "X_dtype": "float32",
		"y_shape": y_np.shape, "y_dtype": "int64",
		"EI_shape": ei_np.shape, "EI_dtype": "int64",
	}
	json.dump(meta, open(mm_dir / "meta.json", "w"))

	Xm = np.memmap(mm_dir / "X.f32.mm", dtype="float32", mode="w+", shape=x_np.shape)
	Xm[:] = x_np.astype("float32")[:]
	ym = np.memmap(mm_dir / "y.i64.mm", dtype="int64", mode="w+", shape=y_np.shape)
	ym[:] = y_np.astype("int64")[:]
	EIm = np.memmap(mm_dir / "EI.i64.mm", dtype="int64", mode="w+", shape=ei_np.shape)
	EIm[:] = ei_np.astype("int64")[:]

	# Save splits for reproducibility
	json.dump({
		"class_list_train": splits["class_list_train"],
		"class_list_val": splits["class_list_val"],
		"class_list_test": splits["class_list_test"],
		"class_dict_train": splits["class_dict_train"],
		"class_dict_val": splits["class_dict_val"],
		"class_dict_test": splits["class_dict_test"],
	}, open(mm_dir / "splits.json", "w"))

	open(mm_dir / "READY", "w").write("ok")


def load_memmap(root: str, dataset: str):
	mm_dir = _mm_root(root, dataset)
	meta = json.load(open(mm_dir / "meta.json"))
	X = np.memmap(mm_dir / "X.f32.mm", dtype=meta["X_dtype"], mode="r", shape=tuple(meta["X_shape"]))
	y = np.memmap(mm_dir / "y.i64.mm", dtype=meta["y_dtype"], mode="r", shape=tuple(meta["y_shape"]))
	EI = np.memmap(mm_dir / "EI.i64.mm", dtype=meta["EI_dtype"], mode="r", shape=tuple(meta["EI_shape"]))
	splits = json.load(open(mm_dir / "splits.json"))

	# Convert back to torch tensors on CPU
	x_t = torch.from_numpy(np.asarray(X))
	y_t = torch.from_numpy(np.asarray(y))
	ei_t = torch.from_numpy(np.asarray(EI))

	# ensure dict keys are int
	cl_tr = splits["class_list_train"]
	cl_va = splits["class_list_val"]
	cl_te = splits["class_list_test"]
	cd_tr = {int(k): v for k, v in splits["class_dict_train"].items()}
	cd_va = {int(k): v for k, v in splits["class_dict_val"].items()}
	cd_te = {int(k): v for k, v in splits["class_dict_test"].items()}

	return x_t, y_t, ei_t, cl_tr, cl_va, cl_te, cd_tr, cd_va, cd_te


