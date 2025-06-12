"""L2 penalty approach to the optimality system."""

import argparse
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq

import natgrad.mlp as mlp
from natgrad.domains import Hyperrectangle
from natgrad.derivatives import laplace

jax.config.update("jax_enable_x64", True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--LM",
    help="Levenberg-Marquardt regularization",
    default=1e-5,
    type=float,
)
parser.add_argument(
    "--iter",
    help="number of iterations",
    default=500,
    type=int,
)
parser.add_argument(
    "--method",
    help="The optimizer",
    default="GN",
    type=str,
)
parser.add_argument(
    "--N_Omega",
    help="number of interior collocation points",
    default=1500,
    type=int,
)
parser.add_argument(
    "--seed",
    help="random seed",
    default=0,
    type=int,
)
parser.add_argument(
    "--N_Gamma",
    help="number of boundary collocation points",
    default=200,
    type=int,
)
args = parser.parse_args()

ITER = args.iter
LM = args.LM
method = args.method
seed = args.seed
N_Omega = args.N_Omega
N_Gamma = args.N_Gamma

print(
    f"PROXIMAL GALERKIN with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}, "
    f"SEED={seed}."
)

# Initialize training components
key = jax.random.PRNGKey(seed)
sampler_key, eval_key, u_key, psi_key, key = jax.random.split(key, num=5)

# distance function to boundary
dist_fct = lambda x: x[0] * (1 - x[0]) * x[1] * (1 - x[1])

# this is a (suboptimal) boundary lift
bdry_lift = lambda x: jnp.where((x[0] > 0), x[0] ** 4, 0.0)

# model for solution u
u_activation = lambda x: jnp.tanh(x)
u_layer_sizes = [2, 32, 1]
u_params = mlp.init_params(u_layer_sizes, u_key)
_u_model = mlp.mlp(u_activation)
u_model = lambda params, x : _u_model(params, x) * dist_fct(x) + bdry_lift(x)
f_u_params, u_unravel = ravel_pytree(u_params)

# model for latent variable psi
psi_activation = lambda x: jnp.tanh(x)
psi_layer_sizes = [2, 16, 1]
psi_params = mlp.init_params(psi_layer_sizes, psi_key)
_psi_model = mlp.mlp(psi_activation)
psi_model = lambda params, x : _psi_model(params, x) * dist_fct(x)
f_psi_params, psi_unravel = ravel_pytree(psi_params)

# put params together, unclear if needed...
params = (u_params, psi_params)
f_params, unravel = ravel_pytree(params)

# collocation points
dim = 2
intervals = [(-1.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
x_Omega = interior.random_integration_points(sampler_key, N=N_Omega)
x_eval = interior.random_integration_points(eval_key, N=10 * N_Omega)

# PDE data and manufactured solutions
alpha = 100.0
u_star = lambda x: jnp.where((x[0] > 0), x[0] ** 4, 0.0)
psi_star = lambda x: jnp.log(u_star(x) + 1e-10)
f = lambda x: jnp.where((x[0] > 0), -12 * x[0] ** 2, 0)
psi_prev = lambda x: 0

# define ingredients for loss functions
def residual_u(u_params, psi_params, x):
    lap_u = laplace(u_model, argnum=1)(u_params, x)
    psi = psi_model(psi_params, x)
    return alpha * lap_u - psi + psi_prev(x) + alpha * f(x)

v_residual_u = jax.vmap(residual_u, (None, None, 0))

def residual_psi(u_params, psi_params, x):
    return u_model(u_params, x) - jnp.exp(psi_model(psi_params, x))

v_residual_psi = jax.vmap(residual_psi, (None, None, 0))

@jax.jit
def loss_fct(u_params, psi_params, X):
    loss_u = 0.5 * jnp.mean(v_residual_u(u_params, psi_params, X) ** 2)
    loss_psi = 0.5 * jnp.mean(v_residual_psi(u_params, psi_params, X) ** 2)
    return loss_u + loss_psi



# Gauss-Newton matrix builders
@jax.jit
def assemble_J(u_params, X):
    def f_grad_lap_u(u_params, x):
        lap_u = lambda u_params, x: laplace(u_model, argnum=1)(u_params, x).squeeze()
        return ravel_pytree(jax.grad(lap_u)(u_params, x))[0]
    return jax.vmap(f_grad_lap_u, (None, 0))(u_params, X)

@jax.jit
def assemble_M(u_params, X):
    def f_fct_u(u_params, x):
        fct_u = lambda u_params, x: u_model(u_params, x).squeeze()
        return ravel_pytree(jax.grad(fct_u)(u_params, x))[0]
    return jax.vmap(f_fct_u, (None, 0))(u_params, X)

@jax.jit
def assemble_M_bar(psi_params, X):
    def f_fct_psi(psi_params, x):
        fct_psi = lambda psi_params, x: psi_model(psi_params, x).squeeze()
        return ravel_pytree(jax.grad(fct_psi)(psi_params, x))[0]
    return jax.vmap(f_fct_psi, (None, 0))(psi_params, X)

@jax.jit
def assemble_M_bar_exp(psi_params, X):
    def f_fct_psi(psi_params, x):
        fct_psi = lambda psi_params, x: psi_model(psi_params, x).squeeze()
        exp_psi = jnp.exp(psi_model(psi_params, x))
        return exp_psi * ravel_pytree(jax.grad(fct_psi)(psi_params, x))[0]
    return jax.vmap(f_fct_psi, (None, 0))(psi_params, X)


@jax.jit
def assemble_gramian(u_params, psi_params, X):
    J = assemble_J(u_params, X)
    M = assemble_M(u_params, X)
    M_bar = assemble_M_bar(psi_params, X)
    M_bar_exp = assemble_M_bar_exp(psi_params, X)

    A = (alpha ** 2 / len(X)) * J.T @ J + (1 / len(X)) * M.T @ M
    B = -(alpha / len(X)) * M_bar.T @ J - (1 / len(X)) * M_bar_exp.T @ M
    C = (1 / len(X)) * M_bar.T @ M_bar + (1 / len(X)) * M_bar_exp.T @ M_bar_exp 
    
    # concat code from ChatGPT
    top = jnp.concatenate([A, B.T], axis=1)
    bottom = jnp.concatenate([B, C], axis=1)
    
    return jnp.concatenate([top, bottom], axis=0)


# error metrics
def l2_error_u(u_params, X):
    return jnp.mean(jax.vmap(lambda x: (u_model(u_params, x) - u_star(x)) ** 2)(X)) ** 0.5
 
def l2_error_psi(psi_params, X):
    return jnp.mean(jax.vmap(lambda x: (psi_model(psi_params, x) - psi_star(x)) ** 2)(X)) ** 0.5

lr = 1e-1

for iteration in range(100_000):

    if method == "GD":
        # autodiff magic
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(u_params, psi_params, x_Omega)

        # param update
        params = jax.tree.map(lambda K, dK: K - lr * dK, params, grads)

        u_params, psi_params = params

        if iteration % 100 == 0:
            print(
                f"Iter {iteration}, loss {loss}, y_error {l2_error_u(u_params, x_eval)}, "
                f"p_error {l2_error_psi(psi_params, x_eval)}"
            )

    if method == "GN":
        # autodiff magic
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(u_params, psi_params, x_Omega)
        f_grads = ravel_pytree(grads)[0]

        # build and regularize the Gramian
        G = assemble_gramian(u_params, psi_params, x_Omega)
        G += 1e-5 * jnp.identity(len(G))

        # compute natural gradient
        f_nat_grad = lstsq(G, f_grads, rcond=-1)[0]
        nat_grads = unravel(f_nat_grad)

        # param update
        lr = 5 * 1e-3
        params = jax.tree.map(lambda K, dK: K - lr * dK, params, nat_grads)
        u_params, psi_params = params

        
        if iteration % 100 == 0:
            print(
                f"Iter {iteration}, loss {loss}, u_error {l2_error_u(u_params, x_eval)}, "
                f"psi_error {l2_error_psi(psi_params, x_eval)}"
            )