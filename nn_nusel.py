from typing import Callable, Any, Tuple
import functools
import numpy as onp
import jax
from jax import numpy as np
import flax
import tqdm

import data
from rf import MLP, ModifiedRPModel
from nn_train import get_optim_spec, build_model
from utils import *

import exputils


parser = exputils.parser('qbdiv-val')
# optim
parser.add_argument('-g_optim', default='adam', type=str)
parser.add_argument('-g_lr', default=4e-3, type=float)
parser.add_argument('-n_iters', default=2000, type=int)
parser.add_argument('-batch_size', default=256, type=int)
parser.add_argument('-n_particles', default=10, type=int)
# model
parser.add_argument('-g_act', default='tanh', type=str)
parser.add_argument('-g_layers', default='50,50,1', type=str)
# model - f
parser.add_argument('-nn_init_scale', default=0.8, type=float)
parser.add_argument('-f_act', default='tanh', type=str)
parser.add_argument('-f_layers', default='50,50,1', type=str)
parser.add_argument('-f_trt_layers', default='16,1', type=str)
add_bool_flag(parser, 'f_factorized', default=False)
add_bool_flag(parser, 'conv_use_gn', default=False)
# model - don't change
parser.add_argument('-f_model', default='linear', type=str)
parser.add_argument('-g_model', default='linear', type=str)
parser.add_argument('-rf_k_scales', default='0.25,1,4', type=str)
parser.add_argument('-n_rfs', default=300, type=int)
# hp range
parser.add_argument('-nu_s', default=0.01, type=float)
parser.add_argument('-nu_e', default=30, type=float)
parser.add_argument('-n_nus', default=10, type=int)
# data
parser.add_argument('-data', default='dgmm-sin', type=str)
parser.add_argument('-N', default=2000, type=int)  # Ntrain
parser.add_argument('-data_corr', default=0.5, type=float)
parser.add_argument('-seed', default=1, type=int)
add_bool_flag(parser, 'hllt_add_endo', default=True)


# === selection of nu ===


def get_val_mse(Dtrain, Dval, model: ModifiedRPModel, nu: float, args):
    """
    return the MSE of predicting f(x)-y|z, averaged over `n_particles` GP prior draws of f.
    """
    assert model.f_nn, NotImplementedError()
    # we can (and should) reuse f_params_init which is unscaled
    # but g_params need to be overriden
    rng = PRNGKeyHolder(jax.random.PRNGKey(args.seed))
    model.g_net, model.g_params_init = model.init_params(
        model.g_factory, rng.gen_key(), model.z_dims, 1., model.g_nn)
    
    N = Dtrain[0].shape[0]
    g_optim = get_optim_spec(args.g_optim, args.g_lr).create(model.g_params_init[0])

    def _loss_fn(params, z_mb, x_mb, y_mb, train, rng=None):
        z, x = model.g_rfe(z_mb), model.f_rfe(x_mb)
        ret = reg = l2_regularizer(params) * nu / N
        gs, f0s = [], []
        for i in range(model.n_particles):
            rng, crf, crg = split_pkey(rng, 3)
            gs.append(model.be_forward('g', i, params[i], z, train, rng=crg))
            f0s.append(model.be_forward('f', i, model.f_params_init[0][i], x, train, rng=crf))

        nmse = 0
        for i in range(model.n_particles):
            nmse += ((gs[i] - (f0s[i] - y_mb)) ** 2).mean() / signal_variance[i]

        stats = {'mnmse': nmse / model.n_particles, 'reg': reg}
        ret += nmse
        return ret, stats

    view_loss = jax.jit(functools.partial(_loss_fn, train=False))
    get_loss_and_grad = jax.value_and_grad(functools.partial(_loss_fn, train=True), has_aux=True)

    @jax.jit
    def _g_step(g_optim, dtuple, lr, rng):
        z_mb, x_mb, y_mb = dtuple
        (loss, stats), grad = get_loss_and_grad(g_optim.target, z_mb, x_mb, y_mb, rng=rng)
        g_optim = g_optim.apply_gradient(grad, learning_rate=lr)
        return g_optim, loss, stats

    def g_step(dtuple):
        nonlocal g_optim, rng
        g_optim, loss, stats = _g_step(g_optim, dtuple, lr, rng.gen_key())
        return loss, stats

    dloader = TensorDataLoader(
        *Dtrain, batch_size=args.batch_size, shuffle=True,
        rng=onp.random.RandomState(args.seed))
    val_dloader = TensorDataLoader(*Dval, batch_size=args.batch_size)
    # get sig variance
    signal_variance = onp.zeros((model.n_particles, ))
    for _, x_mb, y_mb in dloader:
        x = model.f_rfe(x_mb)
        for i in range(model.n_particles):
            f0_i = model.be_forward('f', i, model.f_params_init[0][i], x, train=False)
            ysq_i = ((f0_i - y_mb) ** 2).sum()
            signal_variance[i] += ysq_i
    signal_variance /= N

    # train
    iters_per_ep = (N + args.batch_size - 1) // args.batch_size
    n_epochs = (args.n_iters + iters_per_ep - 1) // iters_per_ep
    lr = args.g_lr
    train_losses = []
    val_stats_accu = StatsAccumulator()
    with tqdm.trange(n_epochs) as trg:
        for c_ep in trg:
            train_loss, stats = traverse_ds(g_step, dloader, has_stats=True)
            val_loss, val_stats = traverse_ds(
                lambda dtuple: view_loss(g_optim.target, *dtuple), val_dloader, has_stats=True)
            stats.update(dict(('val'+k, v) for k, v in val_stats.items()))
            trg.set_postfix(**stats)
            if c_ep > 0 and train_loss + 1e-3 > train_losses[-1]:
                lr *= 0.5
                print(f'decreasing lr to {lr}')
                if lr < 1e-5:
                    print('lr too small. stopping')
                    break
            elif c_ep > 3 and val_loss + 1e-3 > val_stats_accu['_loss'].maximum(s=-3):
                print('early stopping')
                break
            train_losses.append(train_loss)
            val_stats_accu.append(val_stats)

    min_idc = val_stats_accu['_loss'].argmin()
    return val_stats_accu['mnmse'][min_idc]


def main(args):
    (Dtrain, _), _, _ = data.load_data(
        args.data, args.N*3, args.seed, args, split_ratio=2/3)
    (Dtrain, Dval) = data_split(*Dtrain, split_ratio=0.5, rng=onp.random.RandomState(args.seed))
    pkey = jax.random.PRNGKey(args.seed)
    _, _, model = build_model(Dtrain[0], Dtrain[1], args, lam=1., nu=1., pkey=pkey)

    nu_space = log_linspace(args.nu_s, args.nu_e, args.n_nus)
    best = (1e100, None)
    for nu in nu_space:
        vmse = get_val_mse(Dtrain, Dval, model, nu, args)
        print('nu =', nu, 's1_vmse =', vmse)
        if (vmse, nu) < best:
            best = (vmse, nu)
    
    (vmse, nu) = best
    print('Optimal nu =', nu, 's1_vmse =', vmse)


if __name__ == '__main__':
    args = parser.parse_args()
    exputils.preflight(args)
    onp.random.seed(args.seed)
    main(args)
