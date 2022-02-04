"""
Reproduces the image-based demand experiments.
To gather the results:

  cd exp_dir; ls -d * | python /code/exps/gather-image-exp.py

NOTE: 
* you need to ensure that
    * the default arg values in nn_train.py and nn_nusel.py match,
    * G_TRAINING_OPTS contains all argument specific to the validator
* the code is unoptimized and requires 16-20 GB of GPU memory.

The range of the hyperparameters are determined from preliminary runs.
"""

import runner
from runner.utils import _get_timestr, safe_path_str
import logging
import os
import shutil
import sys
import argparse
import numpy as onp


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-nusel', action='store_true', default=True)
parser.add_argument('-no_nusel', action='store_false', dest='nusel')
parser.add_argument('-train', action='store_true', default=True)
parser.add_argument('-no_train', action='store_false', dest='train')
parser.add_argument('-nusel_dir', type=str, default='~/run/qbdiv/nn-nusel_3-25-22-44/')
parser.add_argument('--n_max_gpus', '-ng', type=int, default=4)
parser.add_argument('--n_multiplex', '-nm', type=int, default=1)
parser.add_argument('-ff', action='store_true', default=False)
parser.add_argument('-no_ff', action='store_true', default=True)
parser.add_argument('-Ns', type=str, default='50000')
parser.add_argument('--remove_Ns', '-rNs', type=str, default='')
parser.add_argument('-rhos', type=str, default='0.5')
parser.add_argument('-seed_s', type=int, default=31)
parser.add_argument('-seed_e', type=int, default=34)


def log_linspace(s, e, n):
    ret = onp.exp(onp.linspace(onp.log(s), onp.log(e), n))
    return [int(a*1000)/1000 for a in ret]


def split_list_args(s, typ=float):
    if len(s.strip()) == 0:
        return []
    return list(map(typ, s.split(',')))


args = parser.parse_args()
args.Ns = split_list_args(args.Ns, int)
args.remove_Ns = split_list_args(args.remove_Ns, int)
args.rhos = split_list_args(args.rhos, float)
ff_vals = []
if args.ff:
    ff_vals.append(True)
if args.no_ff:
    ff_vals.append(False)

code_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
param_specs = {
    'f_factorized': runner.BooleanOpt(vals=ff_vals),
    'seed': list(range(31, 32)), 
    'n_particles': [10],
    'batch_size': [100],
    'g_lr': [5e-4, 1e-3, 4e-3, 1e-2],
    'N': args.Ns,
    'data_corr': args.rhos,
    'data': ['hllt-im'], 
    'n_iters': [60000],
    ('g_act', 'f_act'): ['relu'], 
    ('f_layers', 'g_layers'): ['32,1'],
}
G_TRAINING_OPTS = ['g_lr']  # training hps will be passed to main.py
G_SKIP_OPTS = ['n_particles']  # these will not

if args.nusel:
    log_dir = os.path.expanduser('~/run/qbdiv/{}_{}/'.format(os.path.basename(__file__).split('.')[0], _get_timestr()))
else:
    log_dir = os.path.expanduser(args.nusel_dir)

env_pref = f'XLA_PYTHON_CLIENT_MEM_FRACTION={0.5/args.n_multiplex:.3f} OMP_NUM_THREADS=4 '

# we have also experimented with nu \in [0.05, 1]
tasks = runner.list_tasks(
    env_pref + 'python nn_nusel.py -nu_s 1 -nu_e 100 ', param_specs, code_working_dir,
    log_dir + 'prefix_rf')

if args.nusel:
    print('\n'.join([t.cmd for t in tasks]))
    print(tasks[:10])
    print(log_dir)
    r = runner.Runner(
        n_max_gpus=args.n_max_gpus, n_multiplex=args.n_multiplex, n_max_retry=-1)
    r.run_tasks(tasks)

if not args.train:
    sys.exit(0)


best_vmse = {}  # experiment_setting_opts -> g_training_opts

for task in tasks:
    assert os.path.exists(task.log_dir), task.log_dir
    if task.option_dict['seed'] != param_specs['seed'][0]:  # just use a larger n_particles there
        continue
    if task.option_dict['N'] in args.remove_Ns:
        continue
    items = sorted([(k, v) for k, v in task.option_dict.items() if k != 'seed'])
    ckeys = tuple(k for k, v in items if k not in G_TRAINING_OPTS and k not in G_SKIP_OPTS)
    if not 'keys' in locals():
        keys = ckeys
    else:
        assert keys == ckeys
    hash_key = tuple(v for k, v in items if k not in G_TRAINING_OPTS and k not in G_SKIP_OPTS)
    hash_val = tuple(task.option_dict[k] for k in G_TRAINING_OPTS)
    with open(os.path.join(task.log_dir, 'stdout')) as fin:
        ln = fin.readlines()[-1]
        ln0, ln1 = ln.split('s1_vmse =')
        nu = float(ln0.split('nu =')[1])
        # round nu to save some path length (...)
        assert nu > 1e-2
        nu = int(nu*1000) / 1000
        s1_vmse = float(ln1)
    if hash_key not in best_vmse or best_vmse[hash_key][0] > s1_vmse:
        best_vmse[hash_key] = (s1_vmse, nu, hash_val)


param_specs = {
    'seed': list(range(args.seed_s, args.seed_e)),  
    'lr_decay_every': [2560, 5120], # 640, 1280
    'retrain_g_every': [1],
    # selection of lambda will be post hoc
    'n_lams': [1],
    'lam_s': list(log_linspace(0.011, 0.55, 6)), # (0.05, 30, 9)
    'n_particles': [5],
    'snapshot_max': [8],
    ('mode', 'val_mode'): [['qb', 'mean']], #, ['bs', 'mean']],
}

keys = tuple(list(keys) + ['nu'] + G_TRAINING_OPTS)
vals = [list(hkey) + [nu] + list(hval) for hkey, (_, nu, hval) in best_vmse.items()]
print(keys)
print(vals)

param_specs.update({keys: vals})
param_specs.update({
    'f_lr': [1e-2, 5e-3, 1e-3]
})

log_dir = os.path.expanduser('~/run/qbdiv/{}-train_{}/'.format(
    os.path.basename(__file__).split('.')[0], _get_timestr()))

tasks = runner.list_tasks(
    env_pref + 'python nn_train.py -f_model linear -g_model linear ',
    param_specs, code_working_dir, log_dir + 'prefix_rf')

print('\n'.join([t.cmd for t in tasks]))
print(len(tasks))
print(log_dir)

os.makedirs(log_dir, exist_ok=True)
shutil.copyfile(__file__, os.path.join(log_dir, 'script.py'))
with open(os.path.join(log_dir, 'script.py'), 'a') as fout:
    print('#', ' '.join(sys.argv), file=fout)

r = runner.Runner(
    n_max_gpus=args.n_max_gpus, n_multiplex=args.n_multiplex, n_max_retry=-1)
r.run_tasks(tasks)
