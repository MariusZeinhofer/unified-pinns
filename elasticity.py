"""
Implementation of 3d elasticity problem.
Manufactured solution from this FEM paper:

https://www.sciencedirect.com/science/article/pii/S0377042709002349?ref=pdf_download&fr=RR-2&rr=80bdfdaa2d780b51

"""

import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, jacrev
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq
import argparse

from natgrad.domains import Hyperrectangle, HyperrectangleBoundary
import natgrad.mlp as mlp
from natgrad.utility import grid_line_search_factory
from natgrad.derivatives import symgrad, div
from natgrad.gram import gram_factory

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
    "--N_Omega",
    help="number of interior collocation points",
    default=1500,
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
N_Omega = args.N_Omega
N_Gamma = args.N_Gamma

print(f"ELASTICITY with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}")

# random seed for model weigths
seed = 0

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [3, 48, 3]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation)
f_params, unravel = ravel_pytree(params)

# collocation points
dim = 3
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
boundary = HyperrectangleBoundary(intervals)
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=N_Omega)
x_eval = interior.random_integration_points(random.PRNGKey(999), N=10 * N_Omega)
x_Gamma = boundary.random_integration_points(random.PRNGKey(0), N=N_Gamma)


# solution and right-hand side
def u_star(x):
    x_1 = x[0]
    x_2 = x[1]
    x_3 = x[2]
    val = (x_1**2 + 1) * (x_2**2 + 1) * (x_3**2 + 1) * jnp.exp(x_1 + x_2 + x_3)
    return jnp.array([val, val, val])


v_u_star = vmap(u_star, (0))


def C_factory(lam, mu):
    def C(eps):
        return lam * jnp.trace(eps) * jnp.identity(len(eps)) + 2.0 * mu * eps

    return C


C = C_factory(lam=0.5769, mu=0.3846)

# locking regime, nu=0.49
# C = C_factory(lam=0.3356, mu=16.4430)


# residuals
def div_symgrad(u, argnum=0):
    eps = symgrad(u, argnum)

    def C_eps(*args, **kwargs):
        return C(eps(*args, **kwargs))

    return div(C_eps, argnum)


def f(x):
    return -div_symgrad(u_star, argnum=0)(x)


interior_res = lambda params, x: div_symgrad(model, argnum=1)(params, x) + f(x)
v_interior_res = vmap(interior_res, (None, 0))

boundary_res = lambda params, x: model(params, x) - u_star(x)
v_boundary_res = vmap(boundary_res, (None, 0))


# loss function
def interior_loss(params):
    return 1.0 / 2.0 * jnp.mean(v_interior_res(params, x_Omega) ** 2)


def boundary_loss(params):
    return 6.0 * 1.0 / 2.0 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)


@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)


# gramians
gram_int = jit(gram_factory(interior_res))
gram_bdry = jit(gram_factory(boundary_res))

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(lambda x: jnp.sum(jacrev(error)(x) ** 2.0) ** 0.5)
v_u_star_abs_grad = vmap(lambda x: jnp.sum(jacrev(u_star)(x) ** 2.0) ** 0.5)

error_divv = vmap(lambda x: jnp.trace(jacrev(v_error)(x) ** 2) ** 0.5)


def l2_norm(f, x_eval):
    return jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error = l2_norm(v_error, x_eval)
u_l2_norm = l2_norm(v_u_star, x_eval)
h1_error = l2_error + l2_norm(v_error_abs_grad, x_eval)
u_h1_norm = u_l2_norm + l2_norm(v_u_star_abs_grad, x_eval)

print(
    f"Before training: loss: {loss(params)} with error "
    f"L2: {l2_error} and error H1: {h1_error}."
)

# natural gradient descent with line search
for iteration in range(ITER):
    # compute gradient of loss
    grads = grad(loss)(params)
    f_grads = ravel_pytree(grads)[0]

    # assemble gramian
    G_int = gram_int(params, x=x_Omega)
    G_bdry = 6.0 * gram_bdry(params, x=x_Gamma)
    G = G_int + G_bdry

    # Marquardt-Levenberg, LM can also be zero
    Id = jnp.identity(len(G))
    G = jnp.min(jnp.array([loss(params), LM])) * Id + G

    # compute natural gradient
    f_nat_grad = lstsq(G, f_grads, rcond=-1)[0]
    nat_grad = unravel(f_nat_grad)

    # one step of NGD
    params, actual_step = ls_update(params, nat_grad)

    if iteration % 50 == 0:
        # errors
        l2_error = l2_norm(v_error, x_eval)
        h1_error = l2_error + l2_norm(v_error_abs_grad, x_eval)

        print(
            f"NG Iteration: {iteration} with loss: {loss(params)} with error "
            f"L2: {l2_error} and error H1: {h1_error} and step: {actual_step}"
        )

l2_error = l2_norm(v_error, x_eval)
h1_error = l2_error + l2_norm(v_error_abs_grad, x_eval)
print(f"ELASTICITY: loss: {loss(params)} L2: {l2_error} H1: {h1_error}")
