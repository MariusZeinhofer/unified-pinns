"""
Implementation of 3d stokes problem.

Manufactured 3d solution from the paper:
https://arxiv.org/pdf/2208.13540.pdf

"""

import argparse
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, jacrev
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq

from natgrad.domains import Hyperrectangle, HyperrectangleBoundary
import natgrad.mlp as mlp
from natgrad.utility import (
    two_variable_grid_line_search_factory as grid_line_search_factory,
)
from natgrad.utility import flatten_pytrees
from natgrad.derivatives import laplace, div
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

print(
    f"STOKES EXACT BC with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}"
)

# random seed for model weigths
seed = 0

# model with exact bc
activation_u = lambda x: jnp.tanh(x)
layer_sizes_u = [3, 32, 3]
params_u = mlp.init_params(layer_sizes_u, random.PRNGKey(seed))
_model_u = mlp.mlp(activation_u)
model_u = (
    lambda params, x: _model_u(params, x)
    * x[0]
    * (1 - x[0])
    * x[1]
    * (1 - x[1])
    * x[2]
    * (1 - x[2])
)
f_params_u, unravel_u = ravel_pytree(params_u)

# model
activation_p = lambda x: jnp.tanh(x)
layer_sizes_p = [3, 32, 1]
params_p = mlp.init_params(layer_sizes_p, random.PRNGKey(seed))
model_p = mlp.mlp(activation_p)
f_params_p, unravel_p = ravel_pytree(params_u)

# collocation points
dim = 3
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
boundary = HyperrectangleBoundary(intervals)
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=N_Omega)
x_eval = interior.random_integration_points(random.PRNGKey(999), N=10 * N_Omega)
x_Gamma = boundary.random_integration_points(random.PRNGKey(0), N=N_Gamma)


# solution and right-hand side
# (2,) -> (2,)
@jit
def u_star0(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    u_1 = 0.0
    u_2 = ((1 - x) ** 2 * x**2 * (1 - y) ** 2 * y**2) * (4 * z**3 - 6 * z**2 + 2 * z)
    u_3 = ((1 - x) ** 2 * x**2 * (1 - z) ** 2 * z**2) * (4 * y**3 - 6 * y**2 + 2 * y)
    return jnp.array([u_1, u_2, u_3])


v_u_star0 = vmap(u_star0, (0))


@jit
def u_star_bcurl(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    u_1 = (1 - x) * x * (1 - y) ** 2 * y**2 * (1 - z) ** 2 * z**2
    u_2 = 0.0
    u_3 = 0.0
    return jnp.array([u_1, u_2, u_3])


@jit
def u_star(x):
    J = jacrev(u_star_bcurl)(x)
    return jnp.array([J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]])


v_u_star = vmap(u_star, 0)

# assert that diveregnece of the true solution is zero
div_ustar = vmap(lambda x: jnp.trace(jacrev(u_star)(x)))
assert jnp.max(jnp.abs(div_ustar(x_eval))) < 1e-16


@jit
def p_star(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return x * y * z * (1 - x) * (1 - y) * (1 - z)


@jit
def f(x):
    return jnp.array([0.0, 0.0])


f = lambda x: -laplace(u_star)(x) + jacrev(p_star)(x)


# stokes operator residual
def interior_res(params_u, params_p, x):
    return (
        laplace(model_u, argnum=1)(params_u, x)
        + f(x)
        - jacrev(model_p, argnums=1)(params_p, x)
    )


v_interior_res = vmap(interior_res, (None, None, 0))


# divergence residual
def div_res(params_u, x):
    return div(model_u, argnum=1)(params_u, x)


v_div_res = vmap(div_res, (None, 0))


# boundary residual
def boundary_res(params_u, x):
    return model_u(params_u, x) - u_star(x)


v_boundary_res = vmap(boundary_res, (None, 0))


# loss function
def loss(params_u, params_p):
    return (
        0.5 * jnp.mean(v_interior_res(params_u, params_p, x_Omega) ** 2)
        + 0.5 * jnp.mean(v_div_res(params_u, x_Omega) ** 2)
        + 4.0 * 0.5 * jnp.mean(v_boundary_res(params_u, x_Gamma) ** 2)
    )


# define gramians
gram_A_1 = jit(gram_factory(interior_res, argnum_1=0, argnum_2=0))
gram_A_2 = jit(gram_factory(boundary_res, argnum_1=0, argnum_2=0))
gram_A_3 = jit(gram_factory(div_res, argnum_1=0, argnum_2=0))


def gram_A(params_u, params_p, x_Omega, x_Gamma):
    return (
        gram_A_1(params_u, params_p, x=x_Omega)
        + 4.0 * gram_A_2(params_u, x=x_Gamma)
        + gram_A_3(params_u, x=x_Omega)
    )


gram_B = jit(gram_factory(interior_res, argnum_1=1, argnum_2=0))
gram_D = jit(gram_factory(interior_res, argnum_1=1, argnum_2=1))


@jit
def gram(params_u, params_p, x_Omega, x_Gamma):
    A = gram_A(params_u, params_p, x_Omega, x_Gamma)
    B = gram_B(params_u, params_p, x=x_Omega)
    C = jnp.transpose(B)
    D = gram_D(params_u, params_p, x=x_Omega)
    col_1 = jnp.concatenate((A, C), axis=0)
    col_2 = jnp.concatenate((B, D), axis=0)
    return jnp.concatenate((col_1, col_2), axis=1)


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error_u = lambda x: model_u(params_u, x) - u_star(x)
v_error_u = vmap(error_u, (0))
v_error_u_abs_grad = vmap(lambda x: jnp.sum(jacrev(error_u)(x) ** 2.0) ** 0.5)

error_p = lambda x: model_p(params_p, x) - p_star(x)
v_error_p = vmap(error_p, (0))
v_error_p_abs_grad = vmap(lambda x: jnp.sum(jacrev(error_p)(x) ** 2.0) ** 0.5)


def l2_norm(f, x_eval):
    return jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error_u = l2_norm(v_error_u, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)

print(
    f"Before training: loss: {loss(params_u, params_p)} with error "
    f"L2: {l2_error_u} and error H1: {h1_error_u}."
)

for iteration in range(ITER):
    grad_u = grad(loss, 0)(params_u, params_p)
    grad_p = grad(loss, 1)(params_u, params_p)

    gram_matrix = gram(params_u, params_p, x_Omega, x_Gamma)

    # Marquardt-Levenberg
    Id = jnp.identity(len(gram_matrix))
    gram_matrix = jnp.min(jnp.array([loss(params_u, params_p), LM])) * Id + gram_matrix

    flat_combined_grad, retrieve_pytrees = flatten_pytrees(grad_u, grad_p)
    long_flat_nat_grad = lstsq(gram_matrix, flat_combined_grad, rcond=-1)[0]
    nat_grad_u, nat_grad_p = retrieve_pytrees(long_flat_nat_grad)

    # update parameters
    params_u, params_p, actual_step = ls_update(
        params_u, params_p, nat_grad_u, nat_grad_p
    )

    if iteration % 50 == 0:
        l2_error_u = l2_norm(v_error_u, x_eval)
        h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)
        h1_semi_p = l2_norm(v_error_p_abs_grad, x_eval)

        print(
            f"NG Iteration: {iteration} with loss: "
            f"{loss(params_u, params_p)} with error "
            f"L2: {l2_error_u} and error H1: {h1_error_u} and "
            f"H1 semi p {h1_semi_p} "
            f"step: {actual_step}"
        )

l2_error_u = l2_norm(v_error_u, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)
h1_semi_p = l2_norm(v_error_p_abs_grad, x_eval)

print(
    f"STOKES EXACT BC: loss {loss(params_u, params_p)} L2 u {l2_error_u} "
    f" H1 u {h1_error_u} H1 semi p {h1_semi_p}."
)
