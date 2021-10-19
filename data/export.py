import os

import numpy as onp
from scipy.io import savemat
import argparse

from data import dgmm_dgp, modified_dgmm_dgp, sigmoid_dgp, load_data
from utils import data_split


parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, default='/tmp/iv-data')
parser.add_argument('-N', type=int, default=2000)
parser.add_argument('-nonadditive', action='store_true', default=True)
parser.add_argument('-sigmoid', action='store_true', default=False)
parser.add_argument('-hllt', action='store_true', default=False)
parser.add_argument('-hllt_add_endo', action='store_true', default=True)
parser.add_argument('-data_corr', default=0.5, type=float)


def gen_dict(Dtrain, seed):
    Dtrain, Dval = data_split(*Dtrain, split_ratio=0.5, rng=onp.random.RandomState(seed))
    to_dump = {}
    for cat in ['train', 'val']:
        suf = cat[:2]
        z, x, y = locals()['D'+cat]
        to_dump.update({'z'+suf: z, 'x'+suf: x, 'y'+suf: y})
    # to_dump['xte'] = onp.linspace(-4, 4, 200)[:, None]  # deprecated. use va
    # to_dump['fte'] = true_f(to_dump['xte'])
    to_dump['fva'] = true_f(to_dump['xva'])
    to_dump['ftr'] = true_f(to_dump['xtr'])
    return to_dump


args = parser.parse_args()
print(args)
os.makedirs(args.path, exist_ok=True)

if not args.sigmoid and not args.hllt:
    dg_fn = modified_dgmm_dgp if args.nonadditive else dgmm_dgp
    for typ in ['sin', 'abs', 'step', 'linear']:
        print(typ)
        for i in range(10): # 20
            (Dtrain, _), true_f, _ = dg_fn(
                args.N*3, typ=typ, seed=i, split_ratio=2/3, iv_strength=args.data_corr)
            to_dump = gen_dict(Dtrain, i)
            savemat(os.path.join(
                args.path, f'{typ}-{args.nonadditive}-{args.data_corr}-{args.N}-{i}.mat'), to_dump)
elif args.sigmoid:
    for nonadditive in [True, False]:
        print(nonadditive)
        for i in range(10):
            (Dtrain, _), true_f, _ = sigmoid_dgp(
                args.N*3, seed=i, split_ratio=2/3, nonadditive=nonadditive)
            to_dump = gen_dict(Dtrain, i)
            savemat(os.path.join(args.path, f'sigm-{nonadditive}-{args.N}-{i}.mat'), to_dump)
else:
    # the R language is a disaster, so we do the preprocessing here
    def standardize(inp, stats=None):
        if stats is not None:
            mm, ss = stats
        else:
            mm, ss = inp.mean(0), inp.std(0)
        return (inp-mm)/ss, (mm, ss)

    for i in range(10):
        (Dtrain, Dtest), true_f, _ = load_data('hllt', args.N*3, seed=i, args=args, split_ratio=2/3)
        to_dump_tr = gen_dict(Dtrain, i)
        to_dump = {}
        to_dump['ztr'], _ = standardize(to_dump_tr['ztr'])
        to_dump['xtr'], xstats = standardize(to_dump_tr['xtr'])
        to_dump['ytr'] = to_dump_tr['ytr']
        to_dump['xva'], _ = standardize(Dtest[1], xstats)
        to_dump['ftr'] = true_f(to_dump_tr['xtr'])
        to_dump['fva'] = true_f(Dtest[1])
        savemat(os.path.join(args.path, f'inp-hllt-{args.data_corr}-{args.N}-{i}.mat'), to_dump)

