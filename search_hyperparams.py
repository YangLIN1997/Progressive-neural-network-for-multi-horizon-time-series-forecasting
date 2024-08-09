import os
import sys
import logging
import argparse
import multiprocessing
from copy import copy
from itertools import product
from subprocess import check_call
import subprocess
import json
import tqdm

import numpy as np
import utils


logger = logging.getLogger('DeepAR.Searcher')

utils.set_logger('param_search.log')

PYTHON = sys.executable
# PYTHON = sys.path('C/:\Users\yang\Desktop\Code_PhD\venv\Scripts\python.exe')
gpu_ids: list
param_template: utils.Params
args: argparse.ArgumentParser
search_params: dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Sanyo', help='Dataset name')
parser.add_argument('--data-dir', default='data', help='Directory containing the dataset')
parser.add_argument('--model-name', default='param_search', help='Parent directory for all jobs')
# parser.add_argument('--relative-metrics', action='store_true', help='Whether to normalize the metrics by label scales')
parser.add_argument('--gpu-ids', nargs='+', default=[0], type=int, help='GPU ids')
# parser.add_argument('--sampling', action='store_true', help='Whether to do ancestral sampling during evaluation')



def launch_training_job(args_fun):
    '''Launch training of the model with a set of hyperparameters in parent_dir/job_name
    Args:
        search_range: one combination of the params to search
    '''

    search_range = args_fun[0]
    param_template = args_fun[1]
    gpu_ids = args_fun[2]
    args = args_fun[3]
    search_params = args_fun[4]
    model_dir = args_fun[5]

    params = {k: search_params[k][search_range[idx]] for idx, k in enumerate(sorted(search_params.keys()))}
    print(params.items())
    model_param_list = '-'.join('_'.join((k, f'{v:.2f}')) for k, v in params.items())
    model_param = copy(param_template)
    for k, v in params.items():
        setattr(model_param, k, v)

    pool_id, job_idx = multiprocessing.Process()._identity
    gpu_id = gpu_ids[pool_id - 1]

    logger.info(f'Worker {pool_id} running {job_idx} using GPU {gpu_id}')

    # Create a new folder in parent_dir with unique_name 'job_name'
    model_name = os.path.join(model_dir, model_param_list)
    model_input = os.path.join(args.model_name, model_param_list)
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Write parameters in json file
    json_path = os.path.join(model_name, 'params.json')
    model_param.save(json_path)
    logger.info(f'Params saved to: {json_path}')

    # Launch training with this config
    cmd = f'{PYTHON} train.py ' \
        f'--model-name={model_input} ' \
        f'--dataset={args.dataset} ' \
        f'--data-folder={args.data_dir} ' \
        f'--search-hyperparameter="GS" '\
        f'--tqdm={True} '

    subprocess.call(cmd)

        # print(555555,PYTHON)
    # logger.info(cmd)
    # check_call(cmd, shell=True, env={'CUDA_VISIBLE_DEVICES': str(gpu_id),
    #                                  'OMP_NUM_THREADS': '4'})


def start_pool(project_list, processes):
    # print(search_params,processes)
    # for i in project_list:
    #     launch_training_job(i)
    pool = multiprocessing.Pool(processes)
    pool.map(launch_training_job, [(i,param_template, gpu_ids, args, search_params, model_dir) for i in project_list])


def main():
    # Load the 'reference' parameters from parent_dir json file
    global param_template, gpu_ids, args, search_params, model_dir

    args = parser.parse_args()
    model_dir = os.path.join('experiments', args.model_name)
    json_file = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_file), f'No json configuration file found at {args.json}'
    param_template = utils.Params(json_file)

    gpu_ids = args.gpu_ids
    logger.info(f'Running on GPU: {gpu_ids}')

    ND_best = {}
    ND_best['ND_best'] = np.inf
    with open(os.path.join(model_dir,'ND_best.json'), 'w') as f_ND:
        json.dump(ND_best, f_ND, indent=4, ensure_ascii=False)
    # with open(os.path.join(model_dir,'ND_best.json')) as f_ND:
    #     print( json.load(f_ND)['ND_best'])

    # Perform hypersearch over parameters listed below
    search_params =  {
        'lstm_dropout': np.arange(0, 0.401, 0.1, dtype=np.float32).tolist(),
        'lstm_hidden_dim': np.arange(5, 50, 10, dtype=np.int).tolist(),
        'lstm_layers': np.arange(2, 6, 1, dtype=np.int).tolist(),
        'learning_rate': (1e-1 ** np.arange(1,4,1, dtype=np.int)).tolist(),
    }

    # search_params =  {
    #     'learning_rate': (1e-1 ** np.arange(1,3,1, dtype=np.int)).tolist(),
    # }
    # print(search_params)
    keys = sorted(search_params.keys())
    search_range = list(product(*[[*range(len(search_params[i]))] for i in keys]))

    pbar = tqdm.tqdm(total=100)
    start_pool(search_range, len(gpu_ids))
    pbar.close()


if __name__ == '__main__':
    # subprocess.call([PYTHON,'train.py','--dataset=Sanyo','--gridsearch=[True]'])
    #
    # # Launch training with this config
    # cmd = f'{PYTHON} train.py '
    #     # print(555555,PYTHON)
    # logger.info(cmd)
    # check_call(cmd, shell=True, env={'CUDA_VISIBLE_DEVICES': str(0),
    #                                  'OMP_NUM_THREADS': '4'})
    #

    main()
    #
    #
