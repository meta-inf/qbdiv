import jax
from jax import numpy as np
from jax.scipy.linalg import solve_triangular
import numpy as onp

from utils import *
from kernels import MaternKernel, CircularMaternKernel


class DataGen(object):

    def __init__(
            self, ktype,
            F0_REGULARITY = 2,  # b
            ILL_POSEDNESS = 0.5,  # p
            U_VAR = 0.5  # Var[U[0,1]] = 1/12
    ):
        Kernel = {
            'circular': CircularMaternKernel,
            'matern': MaternKernel
        }[ktype]
        Nnys = 5000
        Xnys = np.linspace(0, 1, Nnys)[:, None].astype(np.float64)
        if Kernel == CircularMaternKernel:
            def eigenfuncs(n_k, x):  # ground truth
                ret = []
                for n in range(1, 1+(n_k+1)//2):
                    ret.append(np.cos(2*np.pi*n*x))
                    ret.append(np.sin(2*np.pi*n*x))
                return np.concatenate([
                    np.ones_like(x),
                    np.concatenate(ret, 1) * 2**.5
                ], 1)[:, :n_k]
        else:
            KX_GP_MATERN_ORDER = F0_REGULARITY
            Kx_nys = Kernel(k=KX_GP_MATERN_ORDER, x_train=Xnys)
            evals, evecs = np.linalg.eigh(Kx_nys(Xnys, Xnys))
            def eigenfuncs(n_k, x):  # nystrom
                return Kx_nys(x, Xnys) @ evecs[:, :-n_k-1:-1] / evals[:-n_k-1:-1] * (Nnys**0.5)

        def eval_pdf(grid: np.ndarray, K=10) -> np.ndarray:
            """
            @param grid: 1d array, pdf will be evaluated on grid*grid
            """
            phi_xz = np.ones_like(grid)
            ret = 1 * phi_xz[:, None] * phi_xz[None]
            phi_xz_all = eigenfuncs(K, grid[:, None]).squeeze()  # [n_grid, K]
            phi_xz_all -= phi_xz_all.mean(0, keepdims=True)
            for j in range(1, K):
                lam_j = 0.2 * np.power(1.*j, -ILL_POSEDNESS)
                ret = ret + lam_j * phi_xz_all[:, j-1:j] * phi_xz_all[:, j-1:j].T
            return ret

        def gen_horowitz_data(N, pkey: np.ndarray, grid_size=500, n_bases=400, u_var=U_VAR,
                              f0='fixed'):
            p_xz = eval_pdf(np.linspace(0, 1, grid_size), n_bases)
            cdf_xz = np.cumsum(p_xz, axis=1)

            @jax.jit
            def cond_sample(x, eps):
                # simple linear interpolation
                ix_s = np.clip((x * grid_size).astype('i'), 0, grid_size-1)
                ix_e = np.clip(ix_s+1, 0, grid_size-1)
                cdf_z_given_x = (cdf_xz[ix_s] * (x*grid_size-ix_s) + cdf_xz[ix_e] * (ix_s+1-x*grid_size)) / grid_size
                iz = (cdf_z_given_x<=eps).astype('i').sum()-1
                def get_z_interp(iz):
                    rat = (eps - cdf_z_given_x[iz]) / (cdf_z_given_x[iz+1] - cdf_z_given_x[iz])
                    return (iz + rat + 1) / grid_size
                return jax.lax.cond(
                    iz<grid_size-1,
                    lambda iz: jax.lax.cond(
                        iz>=0,
                        get_z_interp,
                        lambda _: np.zeros((), x.dtype), iz),
                    lambda _: np.ones((), x.dtype), iz)
            
            # draw x ~ U[0, 1]
            pkey, ckey = jax.random.split(pkey)
            Eps = jax.random.uniform(ckey, (N, 2))
            Z, Eps = Eps[:, 0], Eps[:, 1]
            X = jax.vmap(cond_sample)(Z, Eps)
        
            if f0 == 'fixed':
                def f0_(x): return x**3 + (1-x)**2/3*np.cos(x*12)
            else:
                seed = int(f0)
                sob_order = F0_REGULARITY
                fourier_coef = onp.random.RandomState(int(seed)).normal(size=(1, n_bases)) 
                # multiply by sqrt{lambda_i} to get the fourier coef for f0 just outside of H^sob_order.
                # use the properly normalized eigenvalues from the kernel
                fourier_coef *= Kernel(k=sob_order, trunc_order=n_bases, x_train=Xnys).rho**0.5
                # fourier_coef *= (1+onp.arange(n_bases)) ** -((sob_order+1)/2)
                def f0_(x): return (eigenfuncs(n_bases, x) * fourier_coef).sum(1)
        
            b_ = f0_(np.linspace(0, 1, grid_size)[:, None]).mean()
            def f0(x):
                if len(x.shape) == 1:
                    return f0_(x[:, None]) - b_
                return (f0_(x) - b_)[:, None]
            # generate u
            if Kernel == CircularMaternKernel:
                U = np.cos(Eps * 2*np.pi)
            else:
                U = Eps
            U = (U - U.mean()) / U.std() * u_var**0.5
            Y = f0(X[:, None]).squeeze() + U
            return Z[:, None], X[:, None], Y[:, None], f0 #, cond_sample

        vars(self).update(locals())


def gen_horowitz_data(kernel, b, p, u_var, N, grid_size=500, n_bases=400, seed=23, f0_seed=23,
                      split_ratio=0.5):
    ret = DataGen(kernel, b, p, u_var).gen_horowitz_data(
        N, jax.random.PRNGKey(seed), grid_size=grid_size, n_bases=n_bases, f0=f0_seed)
    dall, f0 = ret[:-1], ret[-1]
    return data_split(
        *dall, split_ratio=split_ratio, rng=onp.random.RandomState(seed)), f0, 1.
