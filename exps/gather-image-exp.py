import os, sys, json, shutil
import numpy as np


fls = [a.strip() for a in sys.stdin.readlines() if len(a.strip())>0]

pts = {}

for d in fls:
    log_path = os.path.join(d, 'stdout')
    if not os.path.exists(log_path):
        print(f'{log_path} not exists', file=sys.stderr)
        continue
    with open(log_path) as fin:
        ln = fin.readlines()[-2:]
    if len(ln) < 2 or ln[0].find('qloglh') == -1 or ln[1].find('MSE') == -1:
        print(f'{log_path} not completed', file=sys.stderr)
        continue
    with open(os.path.join(d, 'hps.txt')) as fin:
        seed = json.load(fin)['seed']
    val_loss = float(ln[0].split(' ')[10])
    cf_mse = float(ln[1].split(' ')[3][:6])
    cf_cvg = float(ln[1].split(' ')[7][:-1])
    ciw = float(ln[1].split(' ')[11][:-1])
    if seed not in pts:
        pts[seed] = []
    pts[seed].append((val_loss, cf_mse, cf_cvg, ciw, d))

for seed in pts:
    (val_loss, cf_mse, cf_cvg, ciw, d) = min(pts[seed])
    print(seed, val_loss, cf_mse, cf_cvg, ciw, d)
    shutil.copyfile(os.path.join(d, 'vis.svg'), f'/tmp/opt_{seed}.svg')


