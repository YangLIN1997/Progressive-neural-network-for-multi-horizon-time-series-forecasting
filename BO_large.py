import numpy as np
from train import *

from ax.service.ax_client import AxClient
from copy import copy

import argparse
import json
import os
import sys
import utils

dir = os.getcwd()
PYTHON = sys.executable
gpu_ids: list
param_template: utils.Params
args: argparse.ArgumentParser
search_params: dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Sanyo', help='Dataset name')
parser.add_argument('--data-dir', default='../data', help='Directory containing the dataset')
parser.add_argument('--model-name', default='param_search', help='Parent directory for all jobs')
parser.add_argument('--iterations', default=25, help='Iterations of BO')
parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, help='GPU ids')
parser.add_argument('--restore_ax_client', default=False, help='Whether to restore Axclient')
parser.add_argument('--seed', default=0, help='Set random seed')




def evaluate(parameters):
    # x = np.array([parameters.get(f"x{i + 1}") for i in range(6)])
    # Create a new folder in parent_dir with unique_name 'job_name'

    # parameters['learning_rate'] = 0.1**parameters['learning_rate']
    # parameters['dropout'] = 0.1*parameters['dropout']

    model_param_list = '-'.join('_'.join((k, f'{v:.4f}')) for k, v in parameters.items())
    model_param = copy(param_template)
    for k, v in parameters.items():
        setattr(model_param, k, v)
    model_name = os.path.join(model_dir, model_param_list)
    model_input = os.path.join(args.model_name, model_param_list)
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Write parameters in json file
    json_path = os.path.join(model_name, 'params.json')
    model_param.save(json_path)

    cmd = f'{PYTHON} train.py ' \
        f'--model-name={model_input} ' \
        f'--dataset={args.dataset} ' \
        f'--data-folder={args.data_dir} ' \
        f'--search-hyperparameter="BO" '\
        f'--tqdm={False} '

    # subprocess.call(cmd)
    status=os.system(cmd)
    if status==0:
        best_json_path = os.path.join(model_name, 'metrics_test_best_weights.json')
        with open(best_json_path) as f_metrics:
            metrics_test_best = json.load(f_metrics)
            ND = metrics_test_best['ND']
            rou90 = metrics_test_best['rou90']
        return {"ND": (ND, 0.0), "rou90": (rou90, 0.0)}
    return {"ND": (9999, 0.0), "rou90": (9999, 0.0)}




args = parser.parse_args()
model_dir = os.path.join('experiments', args.model_name)
json_file = os.path.join(model_dir, 'params.json')
assert os.path.isfile(json_file), f'No json configuration file found at {args.json}'
param_template = utils.Params(json_file)

gpu_ids = args.gpu_ids
# logger.info(f'Running on GPU: {gpu_ids}')

ax = AxClient(random_seed=args.seed)
ax.create_experiment(
    name="BO_test_experiment",
    parameters=[
        # {"name": "learning_rate", "type": "range", "value_type": "float","bounds": [1e-3, 1e-2,1e-1]},
        # {"name": "lstm_dropout", "type": "choice", "value_type": "float","values": [0, 0.1, 0.2,0.3,0.4,0.5]},
        {"name": "d_model", "type": "choice", "value_type": "int","values": [8,16,24,48],},
        {"name": "d_ff", "type": "choice", "value_type": "int", "values": [8,16,24,48]},
        {"name": "n_heads", "type": "choice", "value_type": "int","values": [4,8,16,24],},
        {"name": "e_layers", "type": "range", "value_type": "int", "bounds": [2,3,4]},
        {"name": "d_layers", "type": "range", "value_type": "int", "bounds": [2,3,4]},
        # {"name": "alpha", "type": "choice", "value_type": "float", "values": [0.0001,0.001,0.05,0.1]},
        {"name": "dropout", "type": "choice", "value_type": "float", "values": [0,0.1,0.2]},
        #     "log_scale": False,  # Optional, defaults to False.
    ],
    objective_name="ND",
    minimize=True,  # Optional, defaults to False.
    # parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    parameter_constraints=["e_layers >= d_layers"],
    # outcome_constraints=["d_layers <= e_layers"],  # Optional.
)



ND_best = {}
ND_best['ND_best'] = np.inf
with open(os.path.join(model_dir, 'ND_best.json'), 'w') as f_ND:
    json.dump(ND_best, f_ND, indent=4, ensure_ascii=False)
    
if args.restore_ax_client == True:
    ax = AxClient.load_from_json_file(os.path.join(model_dir, 'ax_client.json'))  # For custom filepath, pass `filepath` argument.

for i in range(args.iterations):
    print(f"Running trial {i + 1}/{args.iterations}...")
    parameters, trial_index = ax.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    ax.save_to_json_file(os.path.join(model_dir, 'ax_client.json'))  # For custom filepath, pass `filepath` argument.


best_parameters, values = ax.get_best_parameters()
best_parameters
means, covariances = values
means
print(best_parameters,means)


# best_parameters['learning_rate'] = 0.1 ** best_parameters['learning_rate']
# best_parameters['lstm_dropout'] = 0.1 * best_parameters['lstm_dropout']
model_param = copy(param_template)
for k, v in best_parameters.items():
    setattr(model_param, k, v)
model_name = os.path.join('experiments', 'base_model')
if not os.path.exists(model_name):
    os.makedirs(model_name)
# Write parameters in json file
json_path = os.path.join(model_name, 'params.json')
model_param.save(json_path)