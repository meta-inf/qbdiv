"""
Reproduces the low-dimensional demand experiment using NN-based models.

NOTE: 
* you need to ensure that
    * the default arg values in nn_train.py and nn_nusel.py match,
    * G_TRAINING_OPTS contains all argument specific to the validator
* The occasional JITs will be CPU-intensive, so you may want to use a smaller --n_max_gpus

The range of the hyperparameters are determined from preliminary runs.
"""

from experiments.master import runner
from experiments.master.utils import safe_path_str
from experiments.utils import _get_timestr
import logging
import os
import shutil
import sys
import argparse

logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('-nusel', action='store_true', default=True)
parser.add_argument('-no_nusel', action='store_false', dest='nusel')
parser.add_argument('-nusel_dir', type=str, default=None)
parser.add_argument('-train', action='store_true', default=True)
parser.add_argument('-no_train', action='store_false', dest='train')
parser.add_argument('--n_max_gpus', '-ng', type=int, default=7)
parser.add_argument('--n_multiplex', '-nm', type=int, default=3)
parser.add_argument('-Ns', type=str, default='1000,10000')
parser.add_argument('--remove_Ns', '-rNs', type=str, default='')
parser.add_argument('--remove_rhos', '-rRs', type=str, default='')
parser.add_argument('-remove_act', type=str, default='')
parser.add_argument('-rhos', type=str, default='0.1,0.5')
parser.add_argument('-modes', type=str, default='qb,bs')
parser.add_argument('-seed_s', type=int, default=31)
parser.add_argument('-seed_e', type=int, default=34)
parser.add_argument('-rmk', type=str, default='')


args = parser.parse_args()
exp_base_name = os.path.basename(__file__).split('.')[0]

def spl(a): return [] if a=='' else a.split(',')
args.Ns = list(map(int, spl(args.Ns)))
args.remove_Ns = list(map(int, spl(args.remove_Ns)))
args.rhos = list(map(float, spl(args.rhos)))
args.remove_rhos = list(map(float, spl(args.remove_rhos)))
args.modes = list(map(str.strip, spl(args.modes)))

code_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
param_specs = {
    'seed': list(range(31, 32)), 
    'n_particles': [20],
    'g_lr': [1e-3, 4e-3, 1e-2],
    'N': args.Ns,
    'data_corr': args.rhos,
    'data': ['hllt'],
    'n_iters': [50000],
}
G_TRAINING_OPTS = ['g_lr']
G_IGNORE_OPTS = ['seed', 'n_particles']

if args.nusel:
    log_dir = os.path.expanduser(
        '~/run/qbdiv/{}_{}_{}/'.format(exp_base_name, args.rmk, _get_timestr()))
else:
    log_dir = os.path.expanduser(args.nusel_dir)

env_pref = f'XLA_PYTHON_CLIENT_MEM_FRACTION={0.8/args.n_multiplex:.3f} OMP_NUM_THREADS=4 '

tasks = runner.list_tasks(
    env_pref + 'python nn_nusel.py -nu_s 0.05 -nu_e 1. ', param_specs, code_working_dir,
    log_dir + 'prefix_rf')

if args.nusel:
    print('\n'.join([t.cmd for t in tasks]))
    print(tasks[:10])
    print(len(tasks))
    print(log_dir)
    r = runner.Runner(
        n_max_gpus=args.n_max_gpus, n_multiplex=args.n_multiplex, n_max_retry=-1)
    r.run_tasks(tasks)

if not args.train:
    sys.exit(0)


best_vmse = {}

for task in tasks:
    assert os.path.exists(task.log_dir), task.log_dir
    if task.option_dict['seed'] != param_specs['seed'][0]:  # just use a larger n_particles there
        continue
    if task.option_dict['N'] in args.remove_Ns:
        continue
    if task.option_dict['data_corr'] in args.remove_rhos:
        continue
    if task.option_dict['f_act'] == args.remove_act:
        continue
    items = sorted([(k, v) for k, v in task.option_dict.items() if k not in G_IGNORE_OPTS])
    ckeys = tuple(k for k, v in items if k not in G_TRAINING_OPTS)
    if not 'keys' in locals():
        keys = ckeys
    else:
        assert keys == ckeys
    hash_key = tuple(v for k, v in items if k not in G_TRAINING_OPTS)
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
    'lr_decay_every': [320],
    ('lam_s', 'lam_e', 'n_lams'): [[0.005, 2, 9]],
    'n_particles': [10],
    'mode': args.modes,
    'bs_ratio': [1.],
    'val_mode': ['mean'],
}

keys = tuple(list(keys) + ['nu'] + G_TRAINING_OPTS)
vals = [list(hkey) + [nu] + list(hval) for hkey, (_, nu, hval) in best_vmse.items()]
print(keys)
print(vals)

param_specs.update({keys: vals})
param_specs.update({
    'f_lr': [5e-3, 1e-2]
})

log_dir = os.path.expanduser(
    '~/run/qbdiv/{}-train-{}_{}/'.format(exp_base_name, args.rmk, _get_timestr()))

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
