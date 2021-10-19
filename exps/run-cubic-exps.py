"""
The closed-form posterior-related experiments in the NeurIPS paper.
"""
import logging
import os
import shutil
import sys
import argparse

import runner
from runner.utils import _get_timestr


parser = argparse.ArgumentParser()
parser.add_argument('-hllt', action='store_true', default=False)
parser.add_argument('-ng', '--n_gpus', type=int, default=6)
parser.add_argument('-dry', action='store_true', default=False)
parser.add_argument('-se_s', type=int, default=0)
parser.add_argument('-se_e', type=int, default=20)
args = parser.parse_args()


logging.basicConfig(
        stream=sys.stderr, level=logging.DEBUG,
        format='%(filename)s:%(lineno)s %(levelname)s:%(message)s')

code_working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
param_specs = {
    ('kx', 'kz'): ['matern'],
    ('matern_kx', 'matern_kz'): [3, 5],
    'seed': list(range(args.se_s, args.se_e)),
    'f0': ['sin', 'abs', 'step', 'linear'],
    'fix_strength': [0.5, 0.05]
}
log_dir = os.path.expanduser('~/run/qbdiv/{}_{}/'.format(
    os.path.basename(__file__).split('.')[0], _get_timestr()))

cmd_pref = 'OMP_NUM_THREADS=4 python cubic_sim.py -n_lams 10 -n_cv_large 10 -n_cv 50 -lam_e 5 '
if args.hllt:
    del param_specs['f0']
    del param_specs['fix_strength']
    cmd_pref += '-no_na -data hllt -n_trains 1000,2000 '
else:
    cmd_pref += '-nonadditive -data dgmm -n_trains 200,1000 '

tasks = runner.list_tasks(cmd_pref, param_specs, code_working_dir, log_dir)

# add other kernels
del param_specs[('matern_kx', 'matern_kz')]
param_specs[('kx', 'kz')] = ['linear', 'poly', 'rbf']
tasks += runner.list_tasks(cmd_pref, param_specs, code_working_dir, log_dir)

print('\n'.join([t.cmd for t in tasks]))
print(len(tasks))
print(log_dir)
if args.dry:
    sys.exit(0)

os.makedirs(log_dir, exist_ok=True)
shutil.copyfile(__file__, os.path.join(log_dir, 'script.py'))
with open(os.path.join(log_dir, 'script.py'), 'a') as fout:
    print('#', ' '.join(sys.argv), file=fout)

r = runner.Runner(n_max_gpus=args.n_gpus, n_multiplex=1, n_max_retry=-1)
r.run_tasks(tasks)
