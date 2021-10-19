import os
import pickle
import time

import numpy as onp
from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as np
import matplotlib
matplotlib.use('svg')
from matplotlib import pyplot as plt

import exputils

from kernels import *
from iv import kiv, kiv_hps_selection, krr, bootstrap, BootstrapPredictor
from tsls import ridge2sls
from utils import data_split, log_linspace, split_list_args
from data import simple_nonlin_data, dgmm_dgp, modified_dgmm_dgp, demand_data, load_data


parser = exputils.parser('cubicsim')
parser.add_argument('-seed', default=1, type=int)
# kernel
parser.add_argument('-kx', default='rbf', type=str,
                    choices=['rbf', 'matern', 'sm', 'linear', 'poly', 'circular'],
                    help='kernel for x. sm denotes scale mixture of RBF kernels.')
parser.add_argument('-kx_sm_scales', default='0.25,1,4', type=str,
                    help='for the scale mixture kernel')
parser.add_argument('-kz', default='rbf', type=str,
                    choices=['rbf', 'matern', 'sm', 'linear', 'poly', 'circular'],
                    help='kernel for z')
parser.add_argument('-kz_sm_scales', default='0.25,1,4', type=str)
parser.add_argument('-matern_kx', default=5, type=int, help='order of Matern kernel for x, \
                    multiplied by 2. -1: set to match the ground truth in the GT experiment')
parser.add_argument('-matern_kz', default=5, type=int, help='order of Matern kernel for z, \
                    multiplied by 2. -1: set to match the ground truth in the GT experiment')
parser.add_argument('-poly_k', default=3, type=int, help='order of the polynomial kernels')
parser.add_argument('-n_nystrom', default=-1, type=int, help='Nystrom approximation for k_z. \
                    -1 to disable. Note it may be unstable with finite-rank kernels.')
parser.add_argument('-jitter', default=1e-9, type=float, help='jitter in the Cholesky \
                    factorization for Nystrom. should be < 1/n')
# data
parser.add_argument('-data', default='dgmm', type=str)
parser.add_argument('-data_corr', default=0.5, type=float)  # HLLT will access this
# - variants of the DGMM datasets
parser.add_argument('-nonadditive', action='store_true', default=True)
parser.add_argument('-no_na', action='store_false', dest='nonadditive')
parser.add_argument('-f0', default='sin', type=str,
                    choices=['sin', 'sin_m', 'abs', 'step', 'linear'])
# - the GT dataset
parser.add_argument('-gt_f_seed', default=2, type=int)
parser.add_argument('-gt_b', default=2, type=float)
parser.add_argument('-gt_p', default=0.5, type=float)
parser.add_argument('-gt_u_var', default=0.5, type=float)
parser.add_argument('-gt_trunc_order', default=400, type=int,
                    help='truncation order of the Fourier series')
# 
parser.add_argument('-n_trains', default='200,2000', type=str)
parser.add_argument('-n_cv', default=30, type=int)
parser.add_argument('-n_cv_large', default=10, type=int)
parser.add_argument('-bs_ratio', default=1., type=float, help='subsampling ratio for bootstrap \
                    without replacement. 1 - use bootstrap with replacement')
parser.add_argument('-bs_n_repeats', default=20, type=int)
parser.add_argument('-lam_s', default=0.1, type=float)
parser.add_argument('-lam_e', default=20, type=float)
parser.add_argument('-n_lams', default=10, type=int)
parser.add_argument('-fix_nu', default=-1, type=float)
parser.add_argument('-fix_strength', default=-1, type=float)
parser.add_argument('-plt_ylim', default=-1, type=float)
parser.add_argument('-save_model', action='store_true', default=False)  # These can be huge
add_bool_flag(parser, 'hllt_add_endo', default=True)
add_bool_flag(parser, 'time_only', default=False)
add_bool_flag(parser, 'tsls', default=False)


def visualize(Xtr, Ytr, Xte, Yte, true_f, iv_pred_fn, ols_pred_fn=None, Xte_ax=None):
    """
    NOTE: true_f should only be applied to Xte
    """
    if Xte.shape[1] == 1:
        s, e = onp.percentile(Xte, [2.5, 97.5])
        Xte = np.linspace(s, e, 100)[:, None]

    pred_mean, pred_cov_diag = iv_pred_fn(Xte)
    pred_sd = pred_cov_diag**0.5 * 1.95
    if Xte_ax is None:
        Xte_ax = Xte.squeeze()
    plt.plot(Xte_ax, pred_mean.squeeze(), label='prediction')
    plt.fill_between(Xte_ax, pred_mean.squeeze()-pred_sd, pred_mean.squeeze()+pred_sd, alpha=0.2)

    plt.plot(Xte_ax, true_f(Xte).squeeze(), label='actual')
    if ols_pred_fn is not None:
        plt.plot(Xte_ax, ols_pred_fn(Xte).squeeze(), label='ols', linestyle='--')

    if Xtr is not None and Xtr.shape[-1] == 1:
        msk = np.logical_and(Xtr.squeeze()>Xte[0], Xtr.squeeze()<Xte[-1])
        ss = min(Xtr[msk].shape[0], 1000)
        plt.scatter(Xtr[msk][:ss], Ytr[msk][:ss], s=100/ss, color='gray', label='observations')

    # f0 = true_f(Xte)
    plt.xlim(Xte_ax.min(), Xte_ax.max())


def benchmark_hps_sel(
    dtuple, var_space=None, lam_space=None, n_cv=5, rng=None, n_nystrom=-1,
    KxBase=ScaleMixtureKernel, KzBase=ScaleMixtureKernel, vis=False, vis_path_pref=None,
    Xvis=None, fix_nu=-1, time_only=False):

    if rng is None:
        rng = onp.random.RandomState(23)
    if var_space is None:
        var_space = [1.]
    if lam_space is None:
        lam_space = log_linspace(0.1, 10, 10)

    nys_seed = int(rng.randint(0, 100))
    z_nystrom = NystromSampler(
        jax.random.PRNGKey(nys_seed), mn=n_nystrom) if n_nystrom >= 0 else None

    ((Ztr, Xtr, Ytr), (Zte, Xte, Yte)), true_f = dtuple

    def show_stats(stats, lower=True, scale='linear'):
        stats_to_show = stats - np.min(stats) if lower else stats
        color = [a[0] for a in hps]
        plt.figure(figsize=(7,2), facecolor='w')
        plt.subplot(121)
        plt.xlabel('stats')
        plt.ylabel('counterfactual MSE')
        plt.scatter(stats_to_show, errs, s=4, c=color)
        plt.xscale(scale)
        plt.subplot(122)
        plt.xlabel('stats')
        plt.ylabel('CI coverage')
        plt.scatter(stats_to_show, coverages, s=4, c=color)
        plt.xscale(scale)
        plt.tight_layout()

    def test_predictor(pred_fn):
        pred_mean, pred_cov_diag = pred_fn(Xte)
        r = {}
        r['mse'] = ((pred_mean - true_f(Xte))**2).mean()
        r['l2_cb_rad'] = pred_cov_diag.mean()
        ci = 1.96 * (pred_cov_diag**0.5)[:, None]
        r['ci_coverage'] = (np.abs(pred_mean-true_f(Xte)) <= ci).astype('f').mean()
        r['ci_width'] = ci.mean()
        return dict((k, float(v)) for k, v in r.items())

    if fix_nu <= 0:
        # determine nu.  The optima is independent of the variance of Kx.
        nu_space = log_linspace(1e-2, 1e2, 10)
        statss = 0
        Kz, Kx = KzBase(x_train=Ztr), KxBase(var=1., x_train=Xtr)
        for __ in range(n_cv):
            Dtrain, Dval = data_split(Ztr, Xtr, Ytr, split_ratio=0.5, rng=rng)
            _, stats = kiv_hps_selection(
                Dtrain, Dval, Kz, Kx, nu_space=nu_space, z_nystrom=z_nystrom,
                lam_space=None, return_all_stats=True)
            statss += stats
        nu = nu_space[onp.nanargmin(statss)]
    else:
        Dtrain, Dval = data_split(Ztr, Xtr, Ytr, split_ratio=0.5, rng=rng)
        nu = fix_nu
    print('nu =', nu)

    # Fix an arbitrary train/val split for test
    Dtrain_ref = Dtrain

    if time_only:
        assert len(lam_space) == 1
        # JIT warmup
        Kz, Kx = KzBase(x_train=Dtrain_ref[0]), KxBase(var=1., x_train=Dtrain_ref[1])
        kiv(Dtrain_ref[0], Dtrain_ref[1], Dtrain_ref[2], Kz, Kx, lam_space[0], nu,
            z_nystrom=z_nystrom, jitter=args.jitter)
        t = time.time()
        Kz, Kx = KzBase(x_train=Dtrain_ref[0]), KxBase(var=1., x_train=Dtrain_ref[1])
        kiv(Dtrain_ref[0], Dtrain_ref[1], Dtrain_ref[2], Kz, Kx, lam_space[0], nu,
            z_nystrom=z_nystrom, jitter=args.jitter)
        print('elapsed time:', time.time() - t)
        return

    test_results, hps, pred_fns = [], [], []
    stats_all = None

    # determine var and lambda
    for var in var_space:
        Kz, Kx = KzBase(x_train=Ztr[:3000]), KxBase(var=var, x_train=Xtr[:3000])
        # get val stats, averaged over n_cv replications
        if len(lam_space) > 1:
            for i_cv in range(n_cv):
                (Dtrain, Dval) = data_split(Ztr, Xtr, Ytr, split_ratio=0.5, rng=rng)
                _, _, stats = kiv_hps_selection(
                    Dtrain, Dval, Kz, Kx, nu_space=[nu], lam_space=lam_space,
                    return_all_stats=True, z_nystrom=z_nystrom)
                if i_cv == 0:
                    stats_cur = stats
                else:
                    stats_cur = [sci + si for sci, si in zip(stats_cur, stats)]
        else:
            stats_cur = np.zeros((3, 1))

        if stats_all is None:
            stats_all = stats_cur
        else:
            stats_all = [onp.concatenate([sa_i, sc_i], 0)
                         for sa_i, sc_i in zip(stats_all, stats_cur)]

        # get qlh and counterfactual test stats
        for lam in lam_space:
            pred_lam = kiv(Dtrain_ref[0], Dtrain_ref[1], Dtrain_ref[2], Kz, Kx, lam, nu,
                           z_nystrom=z_nystrom, jitter=args.jitter)
            test_results.append(test_predictor(pred_lam))
            pred_lam.to_onp()  # move predictors out of GPU to save vram
            hps.append((var, lam))
            pred_fns.append(pred_lam)

    coverages, errs = map(onp.array, [
        [a[k] for a in test_results] for k in ['ci_coverage', 'mse']])
    stats_all = list(zip(map(onp.array, stats_all), ['mse', 'gmmv']))
    stats_gmmv = stats_all[1][0]

    if vis:
        def savefig(name):
            if vis_path_pref is not None:
                plt.savefig(vis_path_pref + name + '.png')
            plt.close()

        for i, (stats, name) in enumerate(stats_all):
            print(np.min(stats), hps[onp.nanargmin(stats)])
            show_stats(stats); plt.title(name); savefig('hps-'+name)

        plt.figure(figsize=(3*len(stats_all)-0.5, 2), facecolor='w')
        for i, (stats, name) in enumerate(stats_all):
            var, lam = hps[onp.nanargmin(stats)]
            kiv_pred_fn = pred_fns[onp.nanargmin(stats)]
            plt.subplot(1, len(stats_all), i+1)
            if Xvis is not None:
                visualize(None, None, Xvis, None, true_f, kiv_pred_fn, Xte_ax=Xvis[:, 0])
            else:
                visualize(Dtrain_ref[1], Dtrain_ref[2], Xte, Yte, true_f, kiv_pred_fn)
            # plt.scatter(Xte, Yte, s=.18, marker='+')
        savefig('pred')

    # prepare stats to save
    res = {}
    for crit, name in stats_all:
        idc = onp.nanargmin(crit)
        res[name] = test_results[idc].copy()
        res[name].update({
            'variance': hps[idc][0],
            'lambda': hps[idc][1],
        })

    best_hps_idc = onp.nanargmin(stats_gmmv)
    var, lam = hps[best_hps_idc]
    qb_pred_fn = pred_fns[best_hps_idc]

    if args.bs_n_repeats > 1:
        # bootstrap baselines. Use the reference train-val split
        Kz, Kx = KzBase(x_train=Ztr), KxBase(var=var, x_train=Xtr)
        bs_pred_fn = bootstrap(
            Dtrain_ref[0], Dtrain_ref[1], Dtrain_ref[2], Kz, Kx, lam, nu,
            ratio=args.bs_ratio, n_repeats=args.bs_n_repeats, rng=rng, z_nystrom=z_nystrom,
            jitter=args.jitter)
        res['bs'] = test_predictor(bs_pred_fn)
        bs_pred_fn.to_onp()
        print(res['bs'], res['gmmv'])
    else:
        bs_pred_fn = None
        print(res['gmmv'])

    return res, (qb_pred_fn, bs_pred_fn), locals()


def dgmm_gen_data(Ntrain, typ, seed=1, nonadditive=False, **kw):  # converts onp array to jnp
    fn = dgmm_dgp if not nonadditive else modified_dgmm_dgp
    (Dtrain, Dtest), f0, _ = fn(Ntrain*3, typ, split_ratio=2/3, seed=seed, **kw)
    Dtrain = tuple(map(np.array, Dtrain))  # train and val
    Dtest = tuple(map(np.array, Dtest))
    return ((Dtrain, Dtest), f0)


def visualize_hps_exp(lcs, mode='qb'):
    if mode == 'tsls-bs':
        mode = 'bs'
    ks = lcs['stats_gmmv']
    Dtrain = lcs['Dtrain_ref']
    if Dtrain[1].shape[0] > 1000:
        z_nystrom = NystromSampler(jax.random.PRNGKey(23))
    else:
        z_nystrom = None
    krr_pred_fn = krr(Dtrain[1], Dtrain[2], lcs['Kx'], lcs['lam'], nystrom=z_nystrom)
    pred_fn = lcs[mode + '_pred_fn']
    if 'Xvis' in lcs and lcs['Xvis'] is not None:  # multivariate data, visualize with care
        visualize(None, None, lcs['Xvis'], None, lcs['true_f'], pred_fn, None,
                  Xte_ax=lcs['Xvis'][:, 0])
    else:
        visualize(
            Dtrain[1], Dtrain[2], lcs['Xte'], lcs['Yte'], lcs['true_f'], pred_fn, krr_pred_fn)
    if args.plt_ylim > 0:
        plt.ylim(-args.plt_ylim, args.plt_ylim)


def get_kernel_factory(ktype, kscales, matern_k):
    def get_kx(x_train, **kw):
        if ktype == 'linear':
            mean, sd = x_train.mean(0), x_train.std(0)
            return LinearKernel(inp_stats=(mean, sd), intercept=True)
        elif ktype == 'poly':
            return PolynomialKernel(x_train, k=args.poly_k)
        elif ktype == 'rbf':
            return RBFKernel(x_train=x_train, **kw)
        elif ktype == 'matern':
            return MaternKernel(matern_k, x_train=x_train, **kw)
        elif ktype == 'circular':
            return CircularMaternKernel(
                matern_k, x_train=x_train, trunc_order=args.gt_trunc_order, **kw)
        elif ktype == 'sm':
            kw = kw.copy()
            kw['scales'] = split_list_args(kscales)
            return ScaleMixtureKernel(x_train=x_train, **kw)
    return get_kx


def run_2sls(dtuple, fix_nu, lam_space, n_cv, rng=None):
    if rng is None:
        rng = onp.random.RandomState(23)

    ((Ztr, Xtr, Ytr), (Zte, Xte, Yte)), true_f = dtuple

    if fix_nu < 0:
        kfac = get_kernel_factory('linear', None, None)
        Kz, Kx = kfac(x_train=Ztr), kfac(var=1., x_train=Xtr)
        nu_space = log_linspace(1e-2, 1e2, 10)
        statss = 0
        for __ in range(n_cv):
            Dtrain, Dval = data_split(Ztr, Xtr, Ytr, split_ratio=0.5, rng=rng)
            _, stats = kiv_hps_selection(
                Dtrain, Dval, Kz, Kx, nu_space=nu_space, lam_space=None, return_all_stats=True)
            statss += stats
        nu = nu_space[onp.nanargmin(statss)]
    else:
        nu = fix_nu
    print(nu)

    Dtrain, Dval = data_split(Ztr, Xtr, Ytr, split_ratio=0.5, rng=rng)
    Dtrain_ref = Dtrain  # for viz
    Ntr = Dtrain[0].shape[0]
    _, lam = ridge2sls(Dtrain, Dval, lam_space, nu)
    print(lam)

    assert args.bs_ratio + 1e-3 > 1, NotImplementedError()
    pred_fns = []
    for i in range(args.bs_n_repeats):
        idcs = rng.randint(low=0, high=Ntr, size=(Ntr,))
        Dtrain_i = (Dtrain[0][idcs], Dtrain[1][idcs], Dtrain[2][idcs])
        pred_fn_i, _ = ridge2sls(Dtrain_i, Dval, [lam], nu)
        pred_fns.append(pred_fn_i)
    bs_pred_fn = BootstrapPredictor(pred_fns)

    def test_predictor(pred_fn):
        pred_mean, pred_cov_diag = pred_fn(Xte)
        mse = ((pred_mean - true_f(Xte))**2).mean()
        ci = 1.96 * (pred_cov_diag**0.5)[:, None]
        coverage = (np.abs(pred_mean-true_f(Xte)) <= ci).astype('f').mean()
        return mse, coverage

    mse, cic = test_predictor(bs_pred_fn)
    res = {
        'bs': {'mse': mse, 'ci_coverage': cic},
        'gmmv': {'mse': -1, 'ci_coverage': -1}
    }
    print(res['bs'], res['gmmv'])
    return res, [bs_pred_fn], locals()


def main(args):
    # overwrite args for gt sim
    if args.data == 'gt':
        args.nonadditive = False
        if args.matern_kx < 0:
            args.matern_kx = args.gt_b
        if args.matern_kz < 0:
            args.matern_kz = args.matern_kx + args.gt_p*2
        if args.n_lams == -1:
            args.lam_s, args.lam_e, args.n_lams = args.gt_u_var, args.gt_u_var+1e-3, 1

    lam_space = log_linspace(args.lam_s, args.lam_e, args.n_lams)
    rng = onp.random.RandomState(args.seed)

    kx_fn = get_kernel_factory(args.kx, args.kx_sm_scales, args.matern_kx)
    kz_fn = get_kernel_factory(args.kz, args.kz_sm_scales, args.matern_kz)

    def test_multi_ds(dtuples, disp_names, exp_name, Xvis=None):
        print('============', exp_name, '===========')
        stats_to_save, preds, locals_ = [], [], []
        for dtuple in dtuples:  # ((Ztr, Xtr, Ytr), (Zte, Xte, Yte)), true_fn
            N = dtuple[0][0][0].shape[0]
            n_cv = args.n_cv if N<2000 else args.n_cv_large

            if not args.tsls:
                #
                fix_nu = -1
                if args.fix_nu > 0:
                    fix_nu = args.fix_nu
                    if exp_name[-1] == 'n':
                        N0 = dtuples[0][0][0][0].shape[0]
                        fix_nu = args.fix_nu * (N/N0)**(1/(args.matern_kz+1))
                #
                perf, pred_fns, lcs = benchmark_hps_sel(
                    dtuple, lam_space=lam_space, n_cv=n_cv,
                    KzBase=kz_fn, KxBase=kx_fn, n_nystrom=args.n_nystrom, rng=rng,
                    vis=True, vis_path_pref=os.path.join(args.dir, exp_name), Xvis=Xvis,
                    fix_nu=fix_nu, time_only=args.time_only)
            else:
                perf, pred_fns, lcs = run_2sls(dtuple, args.fix_nu, lam_space, n_cv, rng=rng)
            #
            stats_to_save.append(perf)
            preds.append(pred_fns)
            locals_.append(lcs)

        # visualization
        modes = ['bs', 'qb'] if args.bs_n_repeats > 1 else ['qb']
        if args.tsls:
            modes = [] if args.data == 'hllt-im' else ['tsls-bs']
        for mode in modes:
            plt.figure(figsize=(2.8*len(dtuples)+0.1, 2), facecolor='w')
            for i, disp_name in enumerate(disp_names):
                plt.subplot(1, len(dtuples), i+1)
                visualize_hps_exp(locals_[i], mode=mode)
                if i==0:
                    plt.legend()
                plt.title(disp_name)
            plt.savefig(os.path.join(args.dir, exp_name+'-viz-'+mode+'.svg'))
            plt.close()

            if args.data  == 'gt':
                # dump prediction
                x = onp.linspace(0, 1, 200)[:, None]
                preds_ = []
                for lc in locals_:
                    preds_.append({
                        'Dtrain': lc['Dtrain_ref'],
                        'x_grid': x,
                        'f_pred': tuple(map(onp.asarray, lc['qb_pred_fn'](x))),
                        'f0': onp.asarray(lc['true_f'](x))
                    })
                with open(os.path.join(args.dir, exp_name+'-pred.pkl'), 'wb') as fout:
                    pickle.dump(preds_, fout)

        #
        with open(os.path.join(args.dir, exp_name+'-stats.pkl'), 'wb') as fout:
            pickle.dump(stats_to_save, fout)

        if args.save_model:
            with open(os.path.join(args.dir, exp_name+'-models.pkl'), 'wb') as fout:
                pickle.dump(preds, fout)
        #
        return stats_to_save, preds

    n_trains = split_list_args(args.n_trains, typ=int)
    if args.data != 'dgmm':
        assert not args.nonadditive, NotImplementedError()
        def dgen(n):
            (Dtrain, Dtest), f0, _ = load_data(
                args.data, n*3, args.seed, args, split_ratio=2/3)
            Dtrain = tuple(map(np.array, Dtrain))
            return (Dtrain, Dtest), f0
        # HLLT uses the following dset for visualization
        if args.data == 'hllt':
            emo_fea = np.ones((100, 1)) * 4
            Xvis = np.concatenate([
                np.linspace(0, 10, 100).reshape((-1, 1)), 
                emo_fea,
                np.ones((100, 1)) * 17.5,
            ], -1)
        else:  # univariate data
            Xvis = None
        stats, pred_fns = test_multi_ds(
            [dgen(N) for N in n_trains], [f'N={N}' for N in n_trains], 'varying-n', Xvis=Xvis)
        return

    if args.fix_strength > 0:
        rho = args.fix_strength
    else:
        rho = 0.5
    stats, pred_fns = test_multi_ds(
        [dgmm_gen_data(N, args.f0, nonadditive=args.nonadditive, iv_strength=rho, seed=args.seed)
         for N in n_trains],
        [f'N={N}' for N in n_trains], 'varying-n')
    if args.fix_strength > 0:
        return

    N = n_trains[-1]
    iv_strengths = [0.025, 0.5]
    try:
        stats, pred_fns = test_multi_ds(
            [dgmm_gen_data(N, args.f0, nonadditive=args.nonadditive, iv_strength=s, seed=args.seed) 
             for s in iv_strengths],
            [f'α={s:.2f}' for s in iv_strengths],
            'varying-str')
    except Exception as e:
        import IPython; IPython.embed(); raise 1

    if args.nonadditive:
        return

    stats, pred_fns = test_multi_ds(
        [dgmm_gen_data(N, args.f0, nonadditive=args.nonadditive, discrete_z=True, seed=args.seed) 
         for N in n_trains],
        [f'N={N}' for N in n_trains],
        'discrete')


if __name__ == '__main__':
    args = parser.parse_args()
    exputils.preflight(args)
    onp.random.seed(args.seed)
    main(args)
