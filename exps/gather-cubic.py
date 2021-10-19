import os, sys, pickle, json
import numpy as np
import pandas as pd


def pickle_load(fn):
    with open(fn, 'rb') as fin:
        return pickle.load(fin)


def get_kernel_dispname(hps, suf):
    ktype = hps['k'+suf]
    if ktype not in ['matern', 'poly']:
        return ktype
    if ktype == 'matern': 
        return ktype + '-' + str(hps['matern_k'])
    return ktype + '-' + str(hps['poly_k'])


def get_design(hps):
    if hps['data'] != 'dgmm':
        return hps['data']
    assert hps['nonadditive'], "we don't distinguish na for now"
    return hps['data'] + '-' + hps['f0']


def load_single_exp(path):
    with open(os.path.join(path, 'hps.txt')) as fin:
        hps = json.load(fin)
    assert hps['fix_strength'] > 0
    
    hostname = os.uname().nodename
    kx, kz = get_kernel_dispname(hps, 'x'), get_kernel_dispname(hps, 'z')
    
    Ns = list(map(int, hps['n_trains'].split(',')))
    all_stats = pickle_load(os.path.join(path, 'varying-n-stats.pkl'))
    ret = []
    for n, cur_stats in zip(Ns, all_stats):
        for k, kdisp in [('gmmv', 'qb'), ('bs', 'bs')]:
            stats = cur_stats[k].copy()
            stats['method'] = kdisp
            stats['CI Cvg.'] = float(stats['ci_coverage'])
            stats['MSE'] = float(stats['mse'])
            del stats['ci_coverage'], stats['mse']
            stats['N'] = str(n)
            stats['kx'], stats['kz'] = kx, kz
            stats['design'] = get_design(hps)
            stats['seed'] = hps['seed']
            stats['α'] = hps['fix_strength']
            stats['_src'] = hostname + ':' + path
            if k == 'bs':
                stats['variance'] = ret[-1]['variance']
                stats['lambda'] = ret[-1]['lambda']
            ret.append(stats)
        t = ret[-2]; ret[-2] = ret[-1]; ret[-1] = t

    return ret


def load_exps(root_path):
    ret = []
    assert os.path.exists(os.path.join(root_path, 'script.py'))
    for p in os.listdir(root_path):
        pp = os.path.join(root_path, p)
        if os.path.isdir(pp):
            ret += load_single_exp(pp)
    return pd.DataFrame(ret)


if len(sys.argv) > 2 and os.path.isdir(sys.argv[1]):
    df = load_exps(os.path.expanduser(sys.argv[1]))
    with open(sys.argv[2], 'wb') as fout:
        pickle.dump(df, fout)
else:
    sys.stderr.write('usage: python gather-cubic.py <expdir> <out.pkl>\n')
