import os, sys, pickle, json, argparse
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('svg')
import seaborn as sns
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, default='/tmp/out.pkl', help='pd dataframe dumped by gather-cubic.py')
parser.add_argument('-o', type=str, default='/tmp/', help='root path for output')
args = parser.parse_args()


def pickle_load(fn):
    with open(fn, 'rb') as fin:
        return pickle.load(fin)
    
    
def logical_and(*a):
    r = a[0]
    for aa in a[1:]:
        r = np.logical_and(r, aa)
    return r


def pp(a):
    def ps(s):
        s = f'{s:.3f}'
        if not s.startswith('0'):
            return s[:4]
        return s[1:]
    return ps(a.median()) + ' (' + ps(a.std()) + ')'


def render_kname(kname):
    if kname.startswith('matern'):
        return 'ma'+kname.split('-')[1]
    if kname.startswith('poly'):
        return 'poly'
    return kname[:3]

    
bayesiv_df = pd.DataFrame(pickle_load('/home/ziyu/run/qbdiv/bayesiv.pkl'))
bayesiv_df['design'] = 'dgmm-' + bayesiv_df['design']

df = pickle_load(args.i)
df.N = df.N.astype('i')
df = pd.concat([df, bayesiv_df])
kpred = {'linear': 0, 'poly-3': 1, 'matern-3': 2, 'matern-5': 3, 'rbf': 4}
df['_pres'] = 0*df['MSE']
for k in kpred:
    df['_pres'][df['kx']==k] = kpred[k]
df._pres[df.method=='bayesIV'] = -1
df['_pres'][df.method=='qb'] += 0.5
df['_pres'][df.N==1000] += 0.25
df = df.sort_values('_pres')
knames = [s for s in df.kx.unique() if type(s)==str]


with open(os.path.join(args.o, 'app-tbl-cubic.tex'), 'w') as fout:
    for i, alf in enumerate(reversed(df['α'].unique())):

        first_ = True
        fout.write(r'''
\begin{sidewaystable}
    \centering\scriptsize
    \begin{tabular}{cccccccccccc}
''')

        for design in ['sin', 'abs', 'step', 'linear']: #df.design.unique():
            for N in [200, 1000]:
                if first_:
                    first_ = False
                    print('\\toprule', end=' ', file=fout)
                    head_ = True # print header
                    for kx in knames:
                        for method in ['bs', 'qb']:
                            print(('& ' if not head_ else 'Method & bayesIV & '), method+'-'+render_kname(kx), end=' ', file=fout)
                            head_ = False
                    print('\\\\', file=fout)
                print('\\midrule\\multicolumn{12}{l}{', f"$f_0=\;${design}, $N={N}, \\alpha={alf}$", '} \\\\ \\midrule', file=fout)
                for k in ['MSE', 'CI Cvg.', ('ci_width', 'CI Wid.')]:
                    if isinstance(k, str):
                        vk = k
                    else:
                        k, vk = k
                    # bayesIV
                    dd = df[logical_and(df.design=='dgmm-'+design, df.method=='bayesIV', df.N==N, df['α']==alf)]
                    print(vk, '&', pp(dd[k]), end='', file=fout)
                    for kx in knames:
                        for method in ['bs', 'qb']:
                            dd = df[logical_and(df.design=='dgmm-'+design, df.kx==kx, df.method==method, df.N==N, df['α']==alf)]
                            v = pp(dd[k])
                            print(' &', v, end=' ', file=fout)
                            head_ = False
                    print('\\\\', file=fout)
        print(r'''
        \bottomrule
        \end{tabular}
        \caption{Full results in the 1D simulation, for $\alpha=''' + str(alf) + r'''$}\label{tbl:cubic-sim-''' + str(i) + r'''}
        \end{sidewaystable}''', file=fout)
        
        
def proc(df):
    df = df.copy()
    df1 = df[np.logical_not(df.method == 'bayesIV')].copy()
    df1['method'] = df1['method'] + '-' + df1['kx']
    df1 = df1[np.logical_not(df1['method'].isin(['qb-linear', 'qb-poly-3']))]
    df = pd.concat([df[df.method=='bayesIV'], df1])
    return df

def plot_cic(df, ax, yl=False):
    g = sns.boxplot(y='CI Cvg.', x='N', hue='method', width=0.8, data=df, ax=ax)
    sns.despine(ax=ax)
    plt.ylabel('CI Coverage')
    g.legend_.remove()
    plt.ylim(0.1, 1.01)

def plot_ciw(df, ax, yl=False):
    g = sns.boxplot(y='ci_width', x='N', hue='method', width=0.8, data=df, ax=ax)
    sns.despine(ax=ax)
    plt.ylabel('CI Width')
    g.legend_.remove()

def plot_mse(df, ax):
    sns.despine(ax=ax)
    g = sns.boxplot(y='MSE', x='N', hue='method', width=0.8, data=df, ax=ax)
    plt.ylabel('Normalized MSE')
    plt.yscale('log')
    g.legend_.remove()

    
matplotlib.rc('font', size=11)

for design_ in ['sin', 'abs', 'step', 'linear']:
    design = 'dgmm-'+design_
    fig = plt.figure(figsize=(16, 2.3), facecolor='w')
    for i, alf in enumerate([0.5, 0.05]):
        ax = plt.subplot(1, 6, i+1)
        plot_mse(proc(df[logical_and(df.design==design, df['α']==alf)]), ax=ax)
        if i>0: plt.ylabel(None)
        plt.xlabel(f'N (α={alf})')
        ax = plt.subplot(1, 6, 2+i+1)
        plot_cic(proc(df[logical_and(df.design==design, df['α']==alf)]), ax=ax)
        plt.xlabel(f'N (α={alf})')
        if i>0: plt.ylabel(None)
        ax = plt.subplot(1, 6, 4+i+1)
        plot_ciw(proc(df[logical_and(df.design==design, df['α']==alf)]), ax=ax)
        plt.xlabel(f'N (α={alf})')
        if i>0: plt.ylabel(None)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=9, bbox_to_anchor=(1, 1.13), prop={'size': 12})
    plt.tight_layout()
    plt.savefig(os.path.join(args.o, f'cubic-{design}-stats.pdf'), bbox_inches="tight")
    plt.close()
