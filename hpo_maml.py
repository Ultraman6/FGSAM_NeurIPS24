import argparse
import importlib.util
import os
import sys
from typing import Tuple, List, Set
import optuna
from optuna.samplers import GridSampler
import random
import numpy as np
import torch
import importlib

import memmap_utils as mmu


def _load_module_from_dir(model_dir: str, module_name: str):
	module_path = os.path.join(model_dir, f"{module_name}.py")
	if not os.path.exists(module_path):
		raise FileNotFoundError(f"Cannot find {module_name}.py in {model_dir}")
	spec = importlib.util.spec_from_file_location(module_name, module_path)
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	assert spec.loader is not None
	spec.loader.exec_module(module)
	return module


def _params_dirname(params: dict) -> str:
	def _val_to_str(v):
		if isinstance(v, float):
			return f"{v:g}"
		return str(v)
	parts = [f"{k}={_val_to_str(v)}" for k, v in sorted(params.items())]
	return "__".join(parts)


def set_seed(seed: int = 0) -> None:
	"""Set RNG seeds for reproducibility."""
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def build_objective(model_dir: str, dataset: str, search_space, tune_keys: Set[str], seed: int):
	train_mod = _load_module_from_dir(model_dir, 'train')
	config_mod = _load_module_from_dir(model_dir, 'config')

	def objective(trial: optuna.Trial) -> float:
		cfg = config_mod.config_fs(dataset)

		# Suggest values from a single source of truth (search_space)
		for key, values in search_space.items():
			if key in cfg:
				cfg[key] = trial.suggest_categorical(key, values)

		# speed up trials
		cfg['num_episodes'] = min(cfg.get('num_episodes', 500), 300)
		cfg['num_repeats'] = 1
		cfg['num_meta_val'] = min(cfg.get('num_meta_val', 20), 10)
		cfg['num_meta_test'] = min(cfg.get('num_meta_test', 100), 30)
		cfg['patience'] = min(cfg.get('patience', 10), 8)

		# Save results under: model_dir/dataset/param_combo/
		param_dir = _params_dirname({k: cfg[k] for k in search_space.keys() if k in cfg})
		log_dir = os.path.join(model_dir, dataset, param_dir)
		os.makedirs(log_dir, exist_ok=True)

		# different train APIs return slightly different tuples
		result = train_mod.meta_learning_n_times(dataset, log_dir, cfg, cuda_idx=0)

		# Rename the latest json result to seed-named file inside the param directory
		try:
			json_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.json')]
			if json_files:
				latest = max(json_files, key=os.path.getmtime)
				dst_name = f"{seed}.json"
				dst_path = os.path.join(log_dir, dst_name)
				if os.path.abspath(latest) != os.path.abspath(dst_path):
					os.replace(latest, dst_path)
		except Exception:
			pass
		# Prefer validation metric if available; otherwise test mean acc
		if isinstance(result, tuple):
			if len(result) == 3:
				final_acc, val_acc, _ = result
				return float(val_acc) if val_acc is not None else float(final_acc)
			elif len(result) == 2:
				final_acc, _ = result
				return float(final_acc)
			else:
				return float(result[0])
		return float(result)

	return objective


def _build_grid_space(base_cfg, tune_keys: Set[str]):
	# Define MAML grids
	grids = {
		'lr_finetune': [0.5, 0.1, 0.01, 0.001],
		'lr_meta': [0.05, 0.01, 0.003, 0.001, 0.0001],
		'wd': [0.0, 0.001, 0.0005],
		'dropout': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],
		'rho': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0, 1.2],
		'alpha': [0.5, 0.7, 0.9],
	}
	# Keep only requested keys that actually exist in cfg
	search_space = {k: v for k, v in grids.items() if (k in tune_keys) and (k in base_cfg)}
	return search_space


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', type=str, required=True, help='Path like Meta-GNN, Meta-GNN+FGSAM, Meta-GNN+SAM')
	parser.add_argument('--dataset', type=str, nargs='+', default=['corafull'])
	# removed: --cuda
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--tune', type=str, nargs='+', default=['lr_meta','lr_finetune','wd','dropout','rho','alpha'], help='Which config keys to grid search; only applied if present in config')
	parser.add_argument('--use_ray', action='store_true')
	# new parallelism controls
	parser.add_argument('--num_parallels', type=int, default=2, help='Total parallel Ray trials across all GPUs')
	parser.add_argument('--devices', type=str, nargs='+', default=['0'], help='CUDA device indices to use, e.g., 0 1')
	parser.add_argument('--cpu_per_trial', type=int, default=2)
	args = parser.parse_args()

	# Set RNG seeds before running trials
	set_seed(args.seed)

	# Build grid search space per dataset and run study
	config_mod = _load_module_from_dir(args.model_dir, 'config')
	tune_keys = set(args.tune)
	for dataset in args.dataset:
		base_cfg = config_mod.config_fs(dataset)
		search_space = _build_grid_space(base_cfg, tune_keys)

		# Prebuild memmap if requested and not present
		utils_mod = _load_module_from_dir(args.model_dir, 'utils')
		root = getattr(utils_mod, 'ds_root', '../_data')
		if os.getenv('USE_MEMMAP', '0') == '1' and not mmu.available(root, dataset):
			os.environ['USE_MEMMAP'] = '0'
			x, y, edge_index, cl_tr, cl_va, cl_te, cd_tr, cd_va, cd_te = utils_mod.load_data(dataset, base_cfg['class_split'], root=root)
			mmu.save_memmap(root, dataset, x, y, edge_index, {
				'class_list_train': cl_tr, 'class_list_val': cl_va, 'class_list_test': cl_te,
				'class_dict_train': cd_tr, 'class_dict_val': cd_va, 'class_dict_test': cd_te,
			})
			os.environ['USE_MEMMAP'] = '1'

		if args.use_ray:
			# compute gpu allocation
			visible_devices = ",".join(args.devices)
			os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
			num_gpus = max(1, len(args.devices))
			# fraction = num_gpus / num_parallels
			gpu_per_trial = float(num_gpus) / float(max(1, args.num_parallels))
			gpu_per_trial = float(min(1.0, gpu_per_trial))
			ray = importlib.import_module('ray')
			tune = importlib.import_module('ray.tune')
			if not ray.is_initialized():
				ray.init(ignore_reinit_error=True)
			ray_space = {k: tune.grid_search(v) for k, v in search_space.items()}

			def trainable(cfg_ray):
				try:
					per_fraction = min(0.95, gpu_per_trial)
					torch.cuda.set_per_process_memory_fraction(per_fraction, device=0)
				except Exception:
					pass
				train_mod = _load_module_from_dir(args.model_dir, 'train')
				config_mod_inner = _load_module_from_dir(args.model_dir, 'config')
				cfg = config_mod_inner.config_fs(dataset)
				for key in search_space.keys():
					if key in cfg:
						cfg[key] = cfg_ray[key]
				cfg['num_episodes'] = min(cfg.get('num_episodes', 500), 300)
				cfg['num_repeats'] = 1
				cfg['num_meta_val'] = min(cfg.get('num_meta_val', 20), 10)
				cfg['num_meta_test'] = min(cfg.get('num_meta_test', 100), 30)
				cfg['patience'] = min(cfg.get('patience', 10), 8)

				param_dir = _params_dirname({k: cfg[k] for k in search_space.keys() if k in cfg})
				log_dir = os.path.join(args.model_dir, dataset, param_dir)
				os.makedirs(log_dir, exist_ok=True)

				result = train_mod.meta_learning_n_times(dataset, log_dir, cfg, cuda_idx=0)
				if isinstance(result, tuple):
					if len(result) == 3:
						final_acc, val_acc, _ = result
						score = float(val_acc) if val_acc is not None else float(final_acc)
					elif len(result) == 2:
						score = float(result[0])
					else:
						score = float(result[0])
				else:
					score = float(result)
				from ray import tune as _t
				_t.report(score=score)

			analysis = tune.run(
				trainable,
				config=ray_space,
				resources_per_trial={"cpu": int(args.cpu_per_trial), "gpu": float(gpu_per_trial)},
				local_dir=os.path.join(args.model_dir, dataset, "ray_logs"),
			)
			best_cfg = analysis.get_best_config(metric='score', mode='max')
			best_score = analysis.get_best_trial(metric='score', mode='max').last_result['score']
			print(f'[{dataset}] Best params (Ray):', best_cfg)
			print(f'[{dataset}] Best value (Ray):', best_score)
		else:
			os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(args.devices)
			objective = build_objective(args.model_dir, dataset, search_space, tune_keys, args.seed)
			# Use GridSampler to exhaustively search
			sampler = GridSampler(search_space)
			study = optuna.create_study(direction='maximize', sampler=sampler, study_name=f"maml_{os.path.basename(args.model_dir)}_{dataset}")
			# n_trials is implied by GridSampler's cartesian product
			study.optimize(objective)
			print(f'[{dataset}] Best params:', study.best_params)
			print(f'[{dataset}] Best value:', study.best_value)


if __name__ == '__main__':
	main()
