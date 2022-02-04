import jax
from jax import numpy as np
import numpy as onp

from utils import data_split
from . import demand_data, gtsim_data


def load_data(dname, N, seed, args, split_ratio=0.5):
    """
    return: (dtrain_and_val, dtest), f0, y_sd
            the two data tuples consist of (Z, X, Y)
    """
    if dname.startswith('dgmm'):
        return dgmm_dgp(N, typ=dname.split('-')[1].strip(), seed=seed, split_ratio=split_ratio)
    if dname.startswith('na-dgmm'):
        return modified_dgmm_dgp(
            N, typ=dname.split('-')[2].strip(), seed=seed, split_ratio=split_ratio,
            iv_strength=args.data_corr)
    if dname == 'sigmoid':
        return sigmoid_dgp(N, seed, split_ratio=split_ratio)
    if dname == 'simple':
        return simple_nonlin_data(N, args.data_corr, seed, split_ratio=split_ratio)
    if dname == 'gt':
        return gtsim_data.gen_horowitz_data(
            args.kx, args.gt_b, args.gt_p, args.gt_u_var, N, seed=seed, f0_seed=args.gt_f_seed,
            split_ratio=split_ratio)

    # demand uses a fixed test set, and N corresponds to Ntrain
    if dname in ['hllt', 'div', 'hllt-im', 'div-im']:
        use_im = dname.endswith('im')
        N = int(N * split_ratio)
        return gen_demand_data(
            N, args.data_corr, seed, hllt=dname.startswith('hllt'), use_im=use_im,
            add_endo=args.hllt_add_endo)

    raise NotImplementedError(dname)
    
    
def simple_nonlin_data(N, corr_zx, seed, split_ratio=0.8):
    rng = onp.random.RandomState(seed)
    Z = rng.normal(size=(N, 2))
    U = rng.normal(size=(N, 1))
    X = Z[:, :1] * corr_zx + U * 0.8
    f0 = lambda x: 0.2*x**3 + onp.sin(x*2)
    ff = f0(X)
    s = ff.std()
    b = ff.mean()
    true_f = lambda x: (f0(x) - b) / s
    Y = true_f(X) + U * 2
    return data_split(Z, X, Y, split_ratio=split_ratio, rng=rng), true_f, 1.


def sigmoid_dgp(N, seed, split_ratio=0.5, nonadditive=True):
    rng = onp.random.RandomState(seed)
    ev = rng.multivariate_normal(onp.zeros((2,)), onp.array([[1, .5], [.5, 1]]), size=(N,))
    e, v = ev[:, :1], ev[:, 1:]
    w = rng.normal(size=(N, 1))
    def sigmoid(x):
        return 1 / (1 + onp.exp(-x))
    if nonadditive:  # from kernelIV and sieveIV
        z = sigmoid(w)
        iv_strength = 0.5
        strength = iv_strength*2
        x = sigmoid((strength*w + (2-strength)*v) / ((strength**2+(2-strength)**2)**0.5))
        # x = sigmoid((w + v) / (2**0.5))
        def true_fn(x):
            return onp.log(onp.abs(16*x-8) + 1) * onp.sign(x-0.5)
    else:  # from Wiesenfarth et al, 4.2 (a)
        z = w
        x = z + v
        def true_fn(x):
            return onp.log(onp.abs(x-1) + 1) * onp.sign(x-1)
    y = true_fn(x) + e
    s, b = y.std(), y.mean()
    true_f = lambda x: (true_fn(x) - b) / s
    y = (y-b) / s
    return data_split(z, x, y, split_ratio=split_ratio, rng=rng), true_f, 1.


def dgmm_dgp(N, typ='sin', seed=1, split_ratio=0.8, iv_strength=0.5, discrete_z=False, discretization_level=2):
    rng = onp.random.RandomState(seed)
    # https://github.com/CausalML/DeepGMM/blob/c2a62ed885792145849de98d6cf8fe81cd4185ee/scenarios/toy_scenarios.py#L101
    # NOTE the abs design is still different from theirs
    Z = rng.uniform(-3, 3, size=(N, 2))
    if discrete_z:
        Z = (Z/3*discretization_level).astype('i').astype('f')
    U = rng.normal(size=(N, 1))
    X = Z[:,:1] * iv_strength*2 + U * (1-iv_strength)*2 + rng.normal(0, .1, size=(N, 1))
    f0 = {
        'sin': lambda x: onp.sin(x),
        'sin_m': lambda x: onp.sin(x+0.3) * (2**0.5),
        'abs': lambda x: onp.abs(x),
        'step': lambda x: 1*(x<0) + 2.5*(x>=0),
        'linear': lambda x: x
    }[typ]
    Y = f0(X) + U * 2 + rng.normal(0, .1, size=(N, 1))
    s, b = Y.std(), Y.mean()
    true_f = lambda x: (f0(x) - b) / s
    Y = (Y-b) / s
    return data_split(Z, X, Y, split_ratio=split_ratio, rng=rng), true_f, 1.


def modified_dgmm_dgp(N, typ='sin', seed=1, split_ratio=0.8, iv_strength=0.5, discrete_z=False,
                      discretization_level=2):
    """ Similar to the modifications to the NP design in KIV. """
    rng = onp.random.RandomState(seed)

    ev = rng.multivariate_normal(onp.zeros((2,)), onp.array([[1, .5], [.5, 1]]), size=(N,))
    e, v = ev[:, :1], ev[:, 1:]
    w = rng.normal(size=(N, 1))
    def sigmoid(x):
        return 1 / (1 + onp.exp(-x))
    strength = iv_strength*2
    X = (sigmoid((strength*w + (2-strength)*v) / ((strength**2+(2-strength)**2)**0.5)) - 0.5) * 8
    Z = sigmoid(w)

    if discrete_z:
        Z = (Z/3*discretization_level).astype('i').astype('f')
    f0 = {
        'sin': lambda x: onp.sin(x),
        'sin_m': lambda x: onp.sin(x+0.3) * (2**0.5),
        'abs': lambda x: onp.abs(x),
        'step': lambda x: 1*(x<0) + 2.5*(x>=0),
        'linear': lambda x: x
    }[typ]
    Y = f0(X) + v + rng.normal(0, .1, size=(N, 1))
    s, b = Y.std(), Y.mean()
    true_f = lambda x: (f0(x) - b) / s
    Y = (Y-b) / s
    return data_split(Z, X, Y, split_ratio=split_ratio, rng=rng), true_f, 1.


def gen_demand_data(N, cor, seed, hllt=True, use_im=False, add_endo=False):
    """
    return: (Dtrain, Dtest), true_fn, ysd
    NOTE: when use_im is True, true_fn will take a 3-dim (latent) input, or a 786-dim input
          *only if* it is exactly Xtest. This is sufficient for our NN code
    x = [time; emo; price]
    when add_endo is true, z will be [time; emo; z]
    """
    from .demand_data import one_hot, encode_image
    Ztr, Xtr, Ytr, true_fn, ysd = (hllt_gen_data_ if hllt else div_gen_data_)(
        N, cor, seed, use_im=use_im)
    if add_endo:  # X = [time; emo; price]
        Ztr = onp.concatenate([Xtr[:, :-1], Ztr], -1)
    # gen test
    tax = onp.linspace(0, 10, 20)
    eax = onp.array(list(range(7))) + 1
    pax = onp.linspace(10, 25, 20)
    Xtest = [a.reshape((-1, 1)) for a in onp.meshgrid(tax, eax, pax)]
    Xtest_latent = Xtest.copy()
    if use_im:
        Xtest[1] = encode_image((Xtest[1]-1).squeeze(1), seed, test=True) 
    elif not hllt:
        Xtest[1] = one_hot((Xtest[1]-1).astype('i'), 7)
    Xtest = onp.concatenate(Xtest, -1)
    if use_im:  # hacks true_fn
        f0 = true_fn
        def true_fn_wrapped(X):
            if X.shape[-1] == 3:
                return f0(X)
            else:
                assert X.shape == Xtest.shape and onp.all(onp.abs(Xtest-X) < 1e-5)
                return f0(onp.concatenate(Xtest_latent, -1))
        true_fn = true_fn_wrapped
    return ((Ztr, Xtr, Ytr), (None, Xtest, None)), true_fn, ysd


def hllt_gen_data_(ssz, cor, seed, use_im):
    """
    This corresponds to the "HLLT design" in the KIV (and DualIV) experiments, which removes
    the normalization on both Y and price.
    """
    from .demand_data import demand, one_hot, emocoef
    X, Z, Price, Y, g = demand(ssz, seed, ypcor=cor, use_hllt=True, use_images=use_im)
    # concatenate context and treatment
    if not use_im:
        X = onp.concatenate([
            X[:, :1],  # time
            emocoef(X[:, 1:])[:, None],  # emotion. Following the KIV impl; see sim_HLLT.m (called s therein)
            Price], axis=-1)
    else:
        X = onp.concatenate([X, Price], axis=-1)
    ymean, ysd = Y.mean(), Y.std()  # observable thus valid
    # ysd /= 10
    def true_fn(X):
        assert X.shape[1] == 3
        emo = (X[:, 1] - 1).astype('i')
        P = X[:, 2:]
        X = onp.concatenate([X[:, :1], one_hot(emo, 7)], axis=-1)
        return (g(X, None, P) - ymean) / ysd
    return Z, X, (Y-ymean)/ysd, true_fn, ysd


def div_gen_data_(ssz, cor, seed, use_im):
    """
    This should corresponds to the deepIV repo, or the "H design" in the KIV repo
    """
    from demand_data import demand, one_hot, emocoef
    X, Z, Price, Y, g = demand(ssz, seed, ypcor=cor, use_hllt=False)
    X = onp.concatenate([X[:, :1],
                        X[:, 1:],  # following the KIV impl.  See sim_H.m
                        Price], axis=-1)
    ymean, ysd = Y.mean(), Y.std()
    # ysd /= 10
    def true_fn(X):
        assert X.shape[1] == 9
        P = X[:, -1:]
        X = X[:, :-1]
        return (g(X, None, P) - ymean) / ysd
    return Z, X, (Y-ymean)/ysd, true_fn, ysd
