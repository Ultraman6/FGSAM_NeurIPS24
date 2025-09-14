import argparse
import importlib.util
import os
import sys

import optuna
from optuna.samplers import GridSampler


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


def build_objective(model_dir: str, dataset: str, cuda_idx: int, tune_keys):
	train_mod = _load_module_from_dir(model_dir, 'train')
	config_mod = _load_module_from_dir(model_dir, 'config')

	def objective(trial: optuna.Trial) -> float:
		cfg = config_mod.config_fs(dataset)

		def should_tune(key: str) -> bool:
			return (key in tune_keys) and (key in cfg)

		# non-MAML grids use "lr" (not lr_meta/lr_finetune)
		if should_tune('lr'):
			cfg['lr'] = trial.suggest_categorical('lr', [0.01, 0.005, 0.003, 0.001])
		if should_tune('wd'):
			cfg['wd'] = trial.suggest_categorical('wd', [0.0, 0.001, 0.0005])
		if should_tune('dropout'):
			cfg['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.5])
		if should_tune('rho'):
			cfg['rho'] = trial.suggest_categorical('rho', [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0])
		if should_tune('alpha'):
			cfg['alpha'] = trial.suggest_categorical('alpha', [0.5, 0.7, 0.9])

		# speed up
		cfg['num_episodes'] = min(cfg.get('num_episodes', 1000), 400)
		cfg['num_repeats'] = 1
		cfg['num_meta_val'] = min(cfg.get('num_meta_val', 20), 10)
		cfg['num_meta_test'] = min(cfg.get('num_meta_test', 100), 30)
		cfg['patience'] = min(cfg.get('patience', 10), 8)

		log_dir = os.path.join('log_optuna_nonmaml', f"{os.path.basename(model_dir)}_trial_{trial.number}")
		os.makedirs(log_dir, exist_ok=True)

		result = train_mod.meta_learning_n_times(dataset, log_dir, cfg, cuda_idx=cuda_idx)
		# most GPN variants return (final_acc, val_acc, results)
		if isinstance(result, tuple):
			if len(result) >= 2:
				# prefer validation acc when present
				if len(result) >= 3:
					final_acc, val_acc, _ = result
					return float(val_acc) if val_acc is not None else float(final_acc)
				else:
					final_acc = result[0]
					return float(final_acc)
		return float(result)

	return objective


def _build_grid_space(base_cfg, tune_keys):
	# Define non-MAML grids. Primary lr is "lr".
	grids = {
		'lr': [0.01, 0.005, 0.003, 0.001],
		'wd': [0.0, 0.001, 0.0005],
		'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
		'rho': [0.01, 0.05, 0.1, 0.15, 0.2, 0.5, 0.8, 1.0],
		'alpha': [0.5, 0.7, 0.9],
	}
	search_space = {k: v for k, v in grids.items() if (k in tune_keys) and (k in base_cfg)}
	return search_space


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_dir', type=str, required=True, help='Path like GPN, GPN+FGSAM, GPN+SAM, GPN+FGSAMp, GPN+SALP')
	parser.add_argument('--dataset', type=str, default='corafull')
	parser.add_argument('--cuda', type=int, default=0)
	parser.add_argument('--tune', type=str, nargs='+', default=['lr','wd','dropout','rho','alpha'], help='Which config keys to grid search; only applied if present in config')
	args = parser.parse_args()

	config_mod = _load_module_from_dir(args.model_dir, 'config')
	base_cfg = config_mod.config_fs(args.dataset)
	tune_keys = set(args.tune)
	search_space = _build_grid_space(base_cfg, tune_keys)

	objective = build_objective(args.model_dir, args.dataset, args.cuda, tune_keys)
	sampler = GridSampler(search_space)
	study = optuna.create_study(direction='maximize', sampler=sampler, study_name=f"nonmaml_{os.path.basename(args.model_dir)}_{args.dataset}")
	study.optimize(objective)
	print('Best params:', study.best_params)
	print('Best value:', study.best_value)


if __name__ == '__main__':
	main()
