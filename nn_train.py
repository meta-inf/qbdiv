from typing import Tuple, Any
import pickle
from itertools import chain
import functools
from copy import deepcopy
import os

import jax
from jax import numpy as np
from jax.interpreters import xla
import flax
from flax.core import freeze
import numpy as onp
import tqdm

import exputils

from iv import krr, kiv_hps_selection
from rf import RFExpander, ModifiedRPModel, BSModel, MLP, FactorizedEffectNet, HLLT_MNIST_Model
from cubic_sim import visualize
import kernels
import data
from utils import *


parser = exputils.parser('qbdiv')
# optim
parser.add_argument('-f_lr', default=5e-4, type=float)
parser.add_argument('-g_lr', default=1e-3, type=float)
parser.add_argument('-f_optim', default='adam', type=str)
parser.add_argument('-f_act', default='tanh', type=str)
parser.add_argument('-g_optim', default='adam', type=str)
parser.add_argument('-g_act', default='tanh', type=str)
parser.add_argument('-n_iters', default=2000, type=int)
parser.add_argument('-retrain_g_every', default=2, type=int)
parser.add_argument('-train_g_duplex', default=3, type=int)
parser.add_argument('-batch_size', default=256, type=int)
parser.add_argument('-early_stop_tol', default=8, type=int)
parser.add_argument('-n_warmup_iters', default=3000, type=int)
parser.add_argument('-snapshot_every', default=1, type=int)
parser.add_argument('-snapshot_max', default=-1, type=int) # -1: use snapshot_start_ratio
parser.add_argument('-snapshot_start_ratio', default=0.5, type=float)
parser.add_argument('-lr_decay_every', default=500, type=int)  # in iters
parser.add_argument('-lr_decay_rate', default=0.8, type=float)
parser.add_argument('-lr_decay_tol', default=1e-7, type=float)
# model
parser.add_argument('-n_particles', default=10, type=int)
parser.add_argument('-f_model', default='linear', type=str, help='set to linear to use NN models')
parser.add_argument('-f_layers', default='50,50,1', type=str, help='only used if f_model==linear')
# - deprecated
parser.add_argument('-f_trt_layers', default='16,1', type=str)
add_bool_flag(parser, 'f_factorized', default=False)
add_bool_flag(parser, 'conv_use_gn', default=False)
# - g
parser.add_argument('-g_model', default='linear', type=str)
parser.add_argument('-g_layers', default='50,50,1', type=str)
parser.add_argument('-n_rfs', default=500, type=int)
parser.add_argument('-rf_k_scales', default='0.25,1,4', type=str, help='for scale mixture only')
parser.add_argument('-nn_init_scale', default=0.8, type=float)
parser.add_argument('-mode', default='qb', choices=['qb', 'bs'], type=str)
parser.add_argument('-bs_ratio', default=1., type=float,
                    help='1 - bs with replacement, <1: sample w/o replacement')
parser.add_argument('-val_mode', default='mean', choices=['mean', 'qlh', 'contraction'], type=str)
parser.add_argument('-validate_every', default=2, type=int)
parser.add_argument('--n_val_refresh_epochs', '-n_vr_ep', default=24, type=int)
# hp range
# - for NN g, determine nu by grid search using nusel.py
parser.add_argument('-nu', default=-1, type=float)
# - for kernel. No need to adjust this unless you have extra exogenous regressors
parser.add_argument('-nu_s', default=1e-2, type=float)
parser.add_argument('-nu_e', default=1e2, type=float)
parser.add_argument('-n_nus', default=10, type=int)
# - lambda
parser.add_argument('-lam_s', default=1e-1, type=float)
parser.add_argument('-lam_e', default=30, type=float)
parser.add_argument('-n_lams', default=10, type=int)
# data
parser.add_argument('-data', default='dgmm-sin', type=str)
parser.add_argument('-N', default=2000, type=int)  # Ntrain
parser.add_argument('-data_corr', default=0.5, type=float)
parser.add_argument('-seed', default=1, type=int)
parser.add_argument('-plt_ylim', default=-1, type=float)
add_bool_flag(parser, 'hllt_add_endo', default=True)
add_bool_flag(parser, 'save_model', default=False)


def get_kernel(model_name, scales, x_train):
    scales = split_list_args(scales)
    dct = {
        'rbf': lambda: kernels.RBFKernel(x_train=x_train),
        'sm_rbf': lambda: kernels.ScaleMixtureKernel(
            x_train=x_train, scales=scales, KBase=kernels.RBFKernel)
    }
    if model_name in dct:
        return dct[model_name]()
    else:
        inp_stats = (x_train.mean(0), x_train.std(0))
        return kernels.LinearKernel(inp_stats=inp_stats)


class Validator(object):

    def __init__(self, model: ModifiedRPModel, pkey: np.ndarray, N: int, args: Any):
        # assert model.g_nn
        self.n_val_particles = {
            'mean': 1,
            'qlh': args.n_particles
        }[args.val_mode]
        model.gv_net, model.gv_params_init = model.init_params(
            model.g_factory, pkey, model.z_dims, 1., model.g_nn,
            n_particles=self.n_val_particles)
        model.gv_nn = model.g_nn
        self.model = model
        self.val_mode, self.N = args.val_mode, N
        self.cur_lr = args.g_lr
        self.optim = get_optim_spec(args.g_optim, args.g_lr).create(model.gv_params_init[0])
        def tmp(vp, mp, dtup, train, rng=None):
            return self.get_loss_and_stats(vp, mp, dtup, train, rng=rng)
        self.get_loss_and_grad = jax.jit(
            jax.value_and_grad(tmp, has_aux=True), static_argnums=(3,))

    @functools.partial(jax.jit, static_argnums=(0, 4))
    def get_loss_and_stats(
            self,
            val_params: Any,
            model_params: Any,
            dtuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            train: bool,
            rng = None
        ) -> Tuple[np.ndarray, dict]:
        """
        :param dtuple: data minibatch.
                  NOTE the format is always [z, x, yt, y] regardless of args.mode
        :return: {
            'loss': loss for training the validator,
            'stats': validation loss that hps selection procedure should minimize
        }
        """
        z_mb, x_mb, _, y_mb = dtuple
        assert y_mb.dtype == np.float32
        z, x = self.model.g_rfe(z_mb), self.model.f_rfe(x_mb)
        fs = np.stack([self.model.be_forward('f', i, model_params['f'][i], x, train=False)
                       for i in range(self.model.n_particles)])
        if self.val_mode == 'mean':
            tgt = fs.mean(0) - y_mb
            g = self.model.be_forward('gv', 0, val_params[0], z, train=train, rng=rng)
            mse = ((tgt - g) ** 2).mean()
            stats = {'gmm_mse': (g**2).mean(), 'stats': (g**2).mean()}
        elif self.val_mode == 'qloglh':
            tgt = fs - y_mb  # [NP, BS, 1]
            rngs = list(split_pkey(rng, num=self.n_val_particles+1))
            rng = rngs[-1]
            gs = np.stack([
                self.model.be_forward('gv', i, val_params[i], z, train=train, rng=rngs[i])
                for i in range(self.n_val_particles)])
            mse = ((tgt - gs) ** 2).mean(axis=(1, 2)).sum()
            nqlh = -normal_loglh(gs.mean(0), gs.std(0), np.zeros((gs.shape[1], 1))).mean()
            stats = {'gmm_mse': (gs**2).mean(), 'stats': nqlh}
        else:
            raise NotImplementedError(self.val_mode)
        reg = self.model.nu * l2_regularizer(val_params) / self.N
        loss = mse + reg
        stats.update({'v_mse': mse / self.n_val_particles, 'v_reg': reg})
        return loss, stats

    @functools.partial(jax.jit, static_argnums=(0,))
    def _train_iter(self, optim, model_params, dtuple, rng):
        loss_and_stats, grad = self.get_loss_and_grad(
            optim.target, model_params, dtuple, True, rng=rng)
        return optim.apply_gradient(grad, learning_rate=self.cur_lr), loss_and_stats

    def train_iter(self, model_params, dtuple, rng):
        self.optim, loss_and_stats = self._train_iter(self.optim, model_params, dtuple, rng)
        return loss_and_stats

    def val_iter(self, model_params, dtuple):
        return self.get_loss_and_grad(self.optim.target, model_params, dtuple, False)[0]


def train_validator(model, params, train_dloader, val_dloader, n_epochs, rng, trange=False):
    lr_bak = model.validator.cur_lr
    vtl_losses, vva_losses = [], []
    rg = tqdm.trange(n_epochs, mininterval=2) if trange else DummyContext(range(n_epochs))
    with rg as tr:
        for c_ep in tr:
            rng, c_rng = jax.random.split(rng)
            vtl_losses.append(traverse_ds(
                lambda dtup, rng: model.validator.train_iter(params, dtup, rng),
                val_dloader, has_stats=True, rng=c_rng)[0])
            vva_losses.append(traverse_ds(
                lambda dtup: model.validator.val_iter(params, dtup),
                train_dloader, has_stats=True)[0])
            # print(vtl_losses[-1])
            if c_ep > 0 and vtl_losses[-1] + 1e-3 > vtl_losses[-2]:
                model.validator.cur_lr *= 0.6
                if model.validator.cur_lr < 1e-5:
                    break
            elif c_ep >= 3 and vva_losses[-1] + 1e-3 > onp.max(vva_losses[-4:-1]):
                break
        model.validator.cur_lr = lr_bak
    return c_ep


def validate(model, params, Ztr, Xtr, Ytr, Zva, Xva, Yva, Kz=None, nus=None, args=None,
             rng=None):
    """
    return: (validation stats to maximize, other stats for logging)
    """
    if not model.g_nn and Ztr.shape[0] < 5000:  # exact KRR
        if nus is None:
            nus = log_linspace(1e-1, 10, 8)
        Str = model.predict(params, Xtr)[0] - Ytr  # = Ef - Y
        Fva, _, Fva_all = model.predict(params, Xva, return_all=True)
        Sva = Fva - Yva
        Fva_all = np.transpose(np.squeeze(Fva_all, -1), [1, 0])  # [B, n_particles]
        Sva_all = Fva_all - np.tile(Yva, [1, Fva_all.shape[1]])
        # determine nu using the training set as heldout, similar to ``causal validation''
        errs = []
        for nu in nus:
            pred_nu = krr(Zva, Sva, Kz, nu, cg=True)(Ztr)
            errs.append((mse(pred_nu, Str), nu))
        validator_mse, nu = min(errs)
        validator_nmse = validator_mse / (Str**2).mean()
        # estimate cond exp using nu
        all_preds = krr(Zva, Sva_all, Kz, nu)(Zva)
        preds_mean, preds_sd = all_preds.mean(1), all_preds.std(1)
        if args.val_mode == 'qloglh':
            qloglh = normal_loglh(preds_mean, preds_sd, np.zeros_like(preds_mean))
            stats = -qloglh.mean()
        elif args.val_mode == 'mean':
            stats = (preds_mean**2).mean()
        else:
            raise NotImplementedError(args.val_mode)
        return -stats, StatsDict({
            'gmm_mse': (preds_mean**2).mean(), 'stats': stats, 'v_mse': validator_nmse})
    else:
        train_dloader = TensorDataLoader(Ztr, Xtr, Ytr, Ytr, batch_size=args.batch_size)
        val_dloader = TensorDataLoader(Zva, Xva, Yva, Yva, batch_size=args.batch_size)
        n_eps_used = train_validator(
            model, params, train_dloader, val_dloader, args.n_val_refresh_epochs, rng)
        _, val_stats = traverse_ds(
            lambda dtup: model.validator.val_iter(params, dtup),
            train_dloader, has_stats=True)
        val_stats['val_epochs'] = n_eps_used
        return -val_stats['stats'], val_stats


def get_optim_spec(optim_type: str, lr: float, **kw):
    oname = optim_type.capitalize()
    if hasattr(flax.optim, oname):
        return getattr(flax.optim, oname)(lr, **kw)
    raise NotImplementedError()


def _build_image_model(Ztr, Xtr, args, lam, nu, pkey, bootstrap=False):

    def get_k_rfe(inp, ckey):
        e = np.ones((inp.shape[1] - 2,))
        im_mean = inp[:, 1:-1].mean() * e
        mean = np.concatenate([inp[:, :1].mean(0), im_mean, inp[:, -1:].mean(0)], 0)
        sd = np.concatenate([inp[:, :1].std(0), e, inp[:, -1:].std(0)], 0)
        K = kernels.LinearKernel(inp_stats=(mean, sd))
        rfe = RFExpander(inp.shape[-1], args.n_rfs, K, ckey)
        return K, rfe

    N = Ztr.shape[0]
    pkey, ckey = jax.random.split(pkey)
    Kz, g_rfe = get_k_rfe(Ztr, ckey)
    pkey, ckey = jax.random.split(pkey)
    Kx, f_rfe = get_k_rfe(Xtr, ckey)
    z_dims = x_dims = 2 + 28 * 28
    # model
    f_layers = split_list_args(args.f_layers, typ=int)
    f_factory = lambda: HLLT_MNIST_Model(
        gn=args.conv_use_gn, layers=f_layers, act_type=args.f_act,
        init_scale=args.nn_init_scale)
    g_layers = split_list_args(args.g_layers, typ=int)
    g_factory = lambda: HLLT_MNIST_Model(
        gn=args.conv_use_gn, layers=g_layers, act_type=args.g_act,
        init_scale=args.nn_init_scale)
    pkey, ckey = jax.random.split(pkey)
    model = (BSModel if bootstrap else ModifiedRPModel)(
        f_factory, g_factory, args.n_particles, lam, nu, N=N, x_dims=x_dims, z_dims=z_dims,
        f_rfe=f_rfe, g_rfe=g_rfe, f_nn=len(f_layers)>1, g_nn=len(g_layers)>1, pkey0=ckey)
    return Kz, Kx, model


def build_model(Ztr, Xtr, args, lam, nu, pkey, bootstrap=False):
    if args.data == 'hllt-im':
        return _build_image_model(Ztr, Xtr, args, lam, nu, pkey, bootstrap=bootstrap)

    N = Ztr.shape[0]
    # rfe: z
    Kz = get_kernel(args.g_model, args.rf_k_scales, Ztr)
    pkey, ckey = jax.random.split(pkey)
    g_rfe = RFExpander(Ztr.shape[-1], args.n_rfs, Kz, ckey)
    z_dims = Ztr.shape[-1] if isinstance(Kz, kernels.LinearKernel) else args.n_rfs
    # rfe: x
    Kx = get_kernel(args.f_model, args.rf_k_scales, Xtr)
    pkey, ckey = jax.random.split(pkey)
    f_rfe = RFExpander(Xtr.shape[-1], args.n_rfs, Kx, ckey)
    x_dims = Xtr.shape[-1] if isinstance(Kx, kernels.LinearKernel) else args.n_rfs
    # nn / rf linreg model
    # - f
    f_layers = split_list_args(args.f_layers, typ=int)
    if args.f_factorized:
        assert args.data in ['hllt', 'div'], "f_factorized is only for HLLT"
        assert len(f_layers) > 1 and f_layers[-1] == 1
        f_ctx_layers = f_layers[:-1]
        f_trt_layers = split_list_args(args.f_trt_layers, typ=int)
        f_factory = lambda: FactorizedEffectNet(
            ctx_layers=f_ctx_layers, trt_layers=f_trt_layers, act_type=args.f_act,
            init_scale=args.nn_init_scale)
    else:
        f_factory = lambda: MLP(
            layers=f_layers, act_type=args.f_act, init_scale=args.nn_init_scale)
    # - g
    g_layers = split_list_args(args.g_layers, typ=int)
    g_factory = lambda: MLP(
        layers=g_layers, act_type=args.g_act, init_scale=args.nn_init_scale)
    pkey, ckey = jax.random.split(pkey)

    model = (BSModel if bootstrap else ModifiedRPModel)(
        f_factory, g_factory, args.n_particles, lam, nu, N=N, x_dims=x_dims, z_dims=z_dims,
        f_rfe=f_rfe, g_rfe=g_rfe, f_nn=len(f_layers)>1, g_nn=len(g_layers)>1, pkey0=ckey)
    return Kz, Kx, model


def get_params(st): return {'f': st['f'].target, 'g': st['g'].target}


@functools.partial(jax.jit, static_argnums=(0,))
def _f_step(f_grad_fn, state, dat_tuple, lr, rng):
    (f_loss, _), grads = f_grad_fn(get_params(state), dat_tuple, rng=rng)
    nst = {
        'f': state['f'].apply_gradient(grads['f'], learning_rate=lr),
        'g': state['g']
    }
    return nst, f_loss

@functools.partial(jax.jit, static_argnums=(0,))
def _g_step(f_grad_fn, state, dat_tuple, lr, rng):
    (f_loss, stats), neg_grads = f_grad_fn(get_params(state), dat_tuple, rng=rng)
    grads = jax.tree_map(lambda a: -a, neg_grads['g'])
    nst = {
        'f': state['f'],
        'g': state['g'].apply_gradient(grads, learning_rate=lr)
    }
    return nst, -f_loss, stats


def train(dat_tuple, lam, nu, args):
    rng = PRNGKeyHolder(jax.random.PRNGKey(args.seed))
    # prepare data
    Z, X, Y = dat_tuple
    # - for qb
    Yt = np.tile(Y[:, None], [1, args.n_particles, 1])
    Yt = Yt + jax.random.normal(rng.gen_key(), Yt.shape) * lam**0.5
    # - for bs
    Mk = gen_bs_mask(rng.gen_key(), Z.shape[0], args.bs_ratio, args.n_particles)
    # split data into train and validation. the split is fixed throughout a run
    (Ztr, Xtr, Yt_tr, Ytr, Mtr), (Zva, Xva, Yt_va, Yva, Mva) = data_split(
        Z, X, Yt, Y, Mk, split_ratio=0.5, rng=onp.random.RandomState(args.seed))
    # - dataloader
    if args.mode == 'qb':
        dloader = TensorDataLoader(Ztr, Xtr, Yt_tr, Ytr, batch_size=args.batch_size, shuffle=True)
        val_dloader = TensorDataLoader(
            Zva, Xva, Yt_va, Yva, batch_size=args.batch_size, shuffle=True)
    else:
        dloader = TensorDataLoader(Ztr, Xtr, Ytr, Mtr, batch_size=args.batch_size, shuffle=True)
        val_dloader = TensorDataLoader(Zva, Xva, Yva, Mva, batch_size=args.batch_size, shuffle=True)

    N = Ztr.shape[0]
    iters_per_epoch = (N+args.batch_size-1) // args.batch_size
    lr_decay_every = (args.lr_decay_every+iters_per_epoch-1) // iters_per_epoch
    # build model
    Kz, Kx, model = build_model(
        Z, X, args, lam, nu, rng.gen_key(), bootstrap=(args.mode != 'qb'))
    if model.g_nn or args.N >= 5000:
        model.validator = Validator(model, rng.gen_key(), N, args)
    # optim
    _f_grad_fn = jax.value_and_grad(functools.partial(model.loss_fn, train=True), has_aux=True)
    f_opt_def = get_optim_spec(args.f_optim, args.f_lr)
    g_opt_def = get_optim_spec(args.g_optim, args.g_lr)
    cur_state = {
        'f': f_opt_def.create(model.f_params_init[0]),
        'g': g_opt_def.create(model.g_params_init[0])
    }
    # validation
    view_loss = jax.jit(lambda params, dtuple: model.loss_fn(params, dtuple, train=False))

    def g_step(dtuple, lr, rng):
        nonlocal cur_state
        cur_state, g_loss, stats = _g_step(_f_grad_fn, cur_state, dtuple, lr, rng)
        return g_loss, stats
    #
    def retrain_g(n_iters, g_lr, trange=False):
        # reset optimizer states (momentum etc)
        nonlocal cur_state
        cur_state['g'] = g_opt_def.create(cur_state['g'].target)
        train_losses = []
        val_losses = Accumulator()
        n_epochs = (n_iters+iters_per_epoch-1) // iters_per_epoch
        rg = tqdm.trange(n_epochs, mininterval=0.3) if trange else DummyContext(range(n_epochs))
        with rg as tr:
            for c_ep in tr:
                train_loss, train_stats = traverse_ds(
                    lambda dtuple, rng: g_step(dtuple, g_lr, rng), dloader, has_stats=True,
                    rng=rng.gen_key())
                neg_val_loss, val_stats = traverse_ds(
                    functools.partial(view_loss, get_params(cur_state)),
                    val_dloader, has_stats=True)
                if trange:
                    stats = train_stats.filter('loss').add_prefix('g_train')
                    stats.update(val_stats.filter('loss').add_prefix('g_val'))
                    tr.set_postfix(**stats)
                # check training loss, typically 10^{0-1}
                if len(train_losses) > 0 and train_loss + 1e-3 > train_losses[-1]:
                    g_lr *= 0.5
                    print(f"g: training loss doesn't decrease, decreasing g_lr to {g_lr}")
                    if g_lr < 1e-5:
                        print('g_lr too small. stopping')
                        break
                else:  # check validation loss
                    # ideally should use relative error here
                    val_tol = 1e-4 if args.data.find('dgmm') != -1 else 1e-3
                    if c_ep > 3 and -neg_val_loss + val_tol > val_losses.minimum(s=-3):
                        if trange:
                            print('g: early stop')
                        break
                train_losses.append(train_loss)
                val_losses.append(-neg_val_loss)

    print('pretraining g')
    retrain_g(args.n_warmup_iters, args.g_lr, trange=True)

    if hasattr(model, 'validator'):
        print('pretraining validator')
        _dloader = TensorDataLoader(  # validator uses this fixed format
            Ztr, Xtr, Yt_tr, Ytr, batch_size=args.batch_size, shuffle=True)
        _vloader = TensorDataLoader(
            Zva, Xva, Yt_va, Yva, batch_size=args.batch_size, shuffle=True)
        train_validator(
            model, get_params(cur_state), _dloader, _vloader,
            ceil_div(args.n_warmup_iters, iters_per_epoch), trange=True, rng=rng.gen_key())

    print('main loop')
    f_val_losses = Accumulator()
    best_params = None
    params_trace = []
    n_epochs = ceil_div(args.n_iters, iters_per_epoch)
    f_lr, g_lr = args.f_lr, args.g_lr
    c_itr = 0
    with tqdm.trange(n_epochs, mininterval=0.3) as trg:
        for c_ep in trg:
            ctr = Accumulator()
            for i, dtuple in enumerate(dloader):
                for _ in range(args.train_g_duplex):
                    g_step(dtuple, g_lr, rng.gen_key())
                cur_state, f_loss = _f_step(_f_grad_fn, cur_state, dtuple, f_lr, rng.gen_key())
                ctr.append(f_loss)
                c_itr += 1
                if c_itr % args.lr_decay_every == 0:
                    f_lr *= args.lr_decay_rate
                    g_lr *= args.lr_decay_rate
                    if f_lr < args.lr_decay_tol:
                        print('train: lr too small, stopping')
                        break
            
            if f_lr < args.lr_decay_tol:
                break

            stats = {'train_loss': ctr.average(), 'lr': f_lr}

            if c_ep % args.retrain_g_every == 0:
                retrain_g(iters_per_epoch*2, g_lr)

            if c_ep % args.snapshot_every == 0:
                params_trace.append(get_params(cur_state))
                if args.snapshot_max > 0:
                    params_trace = params_trace[-args.snapshot_max:]

            if c_ep % args.validate_every == 0:
                params = get_params(cur_state)
                qloglh, val_stats = validate(
                    model, params, Ztr, Xtr, Ytr, Zva, Xva, Yva, Kz=Kz, nus=[nu],
                    args=args, rng=rng.gen_key())
                c_val_loss = -qloglh
                stats.update(val_stats.add_prefix('val'))
                trg.set_postfix(**stats)
                if best_params is None or c_val_loss < f_val_losses.minimum():
                    best_params = params
                f_val_losses.append(c_val_loss)
                # stop if there is no improvement in the last early_stop_tol epochs
                if len(f_val_losses.a) > args.early_stop_tol and \
                        f_val_losses.minimum(s=-args.early_stop_tol)-1e-3 > f_val_losses.minimum():
                    print('Early stopping')
                    break

    if args.snapshot_max <= 0:
        s = int(len(params_trace) * args.snapshot_start_ratio)
        e = max(s, len(params_trace) - args.early_stop_tol) + 1
        params_trace = params_trace[s: e]

    return model, (c_val_loss, params), \
        (f_val_losses.minimum(), best_params), params_trace, locals()


def gen_pred_fn(model, params_trace):
    def predict(x):
        all_preds = np.stack([model.predict(p, x, return_all=True)[2] for p in params_trace])
        preds = all_preds.mean(0)  # [n_particles, B, 1]
        preds = np.transpose(np.squeeze(preds, -1), [1, 0])  # [B, n_particles]
        return preds.mean(axis=-1, keepdims=True), preds.std(axis=-1)**2
    return predict


def select_nu_kernel_g_nn_f(
        Dtrain, Dval, model: ModifiedRPModel, Kz: kernels.Kernel, nu_space: onp.ndarray):
    """
    choose nu based on the MSE of predicting f(x)-y|z, averaged over `n_particles` GP prior
    draws of f.
    """
    state_init = {'f': model.f_params_init[0], 'g': model.g_params_init[0]}
    Ztr, Xtr, Ytr = Dtrain
    Zva, Xva, Yva = Dval
    Str = model.predict(state_init, Xtr, return_all=True)[2] - Ytr # [f_i - y]: [NP, BS, 1]
    Sva = model.predict(state_init, Xva, return_all=True)[2] - Yva
    Str, Sva = map(lambda a: a.squeeze(-1).T, (Str, Sva))  # [BS, NP]
    errs = []
    for nu in nu_space:
        pred_nu = krr(Ztr, Str, Kz, nu)(Zva)
        errs.append((float(mse(pred_nu, Sva)), nu))
    print(errs)
    return min(errs)[1]


def main(args):
    import matplotlib
    matplotlib.use('svg')
    from matplotlib import pyplot as plt
    (Dtrain, Dtest), true_fn, y_sd = data.load_data(
        args.data, args.N*3, args.seed, args, split_ratio=2/3)
    Dtrain = tuple(map(jax.device_put, Dtrain))  # train and validation
    Ztr, Xtr, _ = Dtrain

    assert ((args.f_model == 'linear') ^ (len(split_list_args(args.f_layers)) == 1)) and\
        ((args.g_model == 'linear') ^ (len(split_list_args(args.g_layers)) == 1)), \
        'either kernel or NN models should be used'

    # determine nu
    if args.g_model != 'linear' and args.nu <= 0:
        assert not args.data.endswith('im'), "kernel construction below will be incorrect"
        Nk = max(Ztr.shape[0], 5000)  # estimating median dist doesn't require a lot of samples
        Kz = get_kernel(args.g_model, args.rf_k_scales, Ztr[:Nk])
        Kx = get_kernel(args.f_model, args.rf_k_scales, Xtr[:Nk])
        nu_space = log_linspace(args.nu_s, args.nu_e, args.n_nus)
        if args.f_model != 'linear':
            stats = 0
            for _ in range(10):
                _dtrain, _dheldout = data_split(*Dtrain, split_ratio=0.5)
                _, ss = kiv_hps_selection(
                    _dtrain, _dheldout, Kz, Kx, nu_space=nu_space, return_all_stats=True)
                stats += ss
            nu = nu_space[onp.nanargmin(stats)]
        else:
            _dtrain, _dheldout = data_split(*Dtrain, split_ratio=0.5)
            _, _, _model = build_model(
                Ztr[:Nk], Xtr[:Nk], args, lam=1., nu=1., pkey=jax.random.PRNGKey(args.seed))
            nu = select_nu_kernel_g_nn_f(_dtrain, _dheldout, _model, Kz, nu_space)
        print('nu =', nu)
    else:
        assert args.nu > 0, "nu should be determined with nusel.py prior to training"
        nu = args.nu

    # determine lam
    cpu = jax.devices('cpu')[0]
    best = (1e100, 1, None)
    for lam in log_linspace(args.lam_s, args.lam_e, args.n_lams):
        xla._xla_callable.cache_clear()
        print('lam =', lam)
        model, _, (best_nqlh, best_params), params_trace, _locals = train(Dtrain, lam, nu, args)
        del _locals
        # save vmem. NOTE: model also contains on-device arrays, but it's smaller than trace
        (best_params, params_trace) = jax.tree_map(
            functools.partial(jax.device_put, device=cpu), (best_params, params_trace))
        best = min(best, (best_nqlh, lam, (best_params, params_trace, model)))

    (best_nqlh, lam, (best_params, params_trace, model)) = best
    # move model back to gpu for prediction
    (best_params, params_trace) = jax.tree_map(jax.device_put, (best_params, params_trace))
    print(f'Optimal nu = {nu:.3f}, lam = {lam:.3f}, neg qloglh = {best_nqlh:.5f}')
    to_dump = {
        'qlh': -best_nqlh, 'lam': lam, 'nu': nu,
        'best_params': best_params, 'trace': params_trace, 'args': args,
        'model_dump': model.dump()
    }
    if args.save_model:
        with open(os.path.join(args.dir, 'model.pkl'), 'wb') as fout:
            pickle.dump(to_dump, fout)

    pred_fn = gen_pred_fn(model, params_trace)
    f_test = true_fn(Dtest[1])
    #
    f_pred, f_pred_var = pred_fn(Dtest[1])
    cf_nmse = mse(f_pred, f_test)
    cf_cic = ci_coverage(f_test, f_pred, f_pred_var[:,None]**0.5)
    ciw = (f_pred_var[:,None]**0.5).mean() * 1.96
    #
    pred_fn_single = gen_pred_fn(model, [best_params])
    f_pred_single, f_pred_var_single = pred_fn_single(Dtest[1])
    cf_nmse_single = mse(f_pred_single, f_test)
    cf_cic_single = ci_coverage(f_test, f_pred_single, f_pred_var_single[:,None]**0.5)
    ciw_single = (f_pred_var_single[:,None]**0.5).mean() * 1.96
    #
    print(f'Counterfactual MSE: avg {cf_nmse}, best {cf_nmse_single}, cic {cf_cic}, cics {cf_cic_single}, ciw {ciw}, ciws {ciw_single}',
          f'(y_sd: {y_sd}, log unnormalized: {np.log(y_sd**2 * cf_nmse) / np.log(10):.3f})')
    plt.figure(figsize=(2.8, 2), facecolor='w')
    if args.data in ['hllt', 'div', 'hllt-im', 'div-im']:
        from demand_data import one_hot, get_images
        emo_fea = emo_fea_latent = np.ones((100, 1)) * 4
        if args.data == 'div':
            emo_fea = one_hot((emo_fea - 1).astype('i'), 7)
        elif args.data == 'hllt-im':
            emo_fea = get_images(emo_fea[0]-1, emo_fea.shape[0], seed=args.seed, testset=True)
        Xvis = np.concatenate([
            np.linspace(0, 10, 100).reshape((-1, 1)), 
            emo_fea,
            np.ones((100, 1)) * 17.5,
        ], -1)
        # Wrap true_fn to return f(Xvis). recall true_fn always takes the 3-dim input
        xv = np.concatenate([Xvis[:, :1], emo_fea_latent, Xvis[:, -1:]], -1)
        true_fn_ = lambda _: true_fn(xv)
        visualize(None, None, Xvis, None, true_fn_, pred_fn, Xte_ax=Xvis[:, 0])
    else:
        visualize(Dtrain[1], Dtrain[2], Dtest[1], Dtest[2], true_fn, pred_fn)
        true_fn_, Xvis = true_fn, Dtest[1]
    # plt.title(f'N={args.N}, lam={lam:.2f}')
    if args.plt_ylim > 0:
        plt.ylim(-args.plt_ylim, args.plt_ylim)
    plt.savefig(os.path.join(args.dir, 'vis.svg'))
    to_dump = {
        'xvis': Xvis,
        'f0': true_fn_(Xvis),
        'fpred': pred_fn(Xvis),
        'fpred_single': pred_fn_single(Xvis),
    }
    with open(os.path.join(args.dir, 'pred.pkl'), 'wb') as fout:
        pickle.dump(to_dump, fout)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.f_model == 'linear' and args.g_model == 'linear':
        from jax.config import config
        config.update("jax_enable_x64", False)
    exputils.preflight(args)
    onp.random.seed(args.seed)
    main(args)
