"""
Inverse problem, source recovery -- PDE constrained optimization

The concrete example is:

min E(u,f) = ||laplace(u) + f||^2 + ||u_d - u||^2 + ||u - g||^2_Gamma

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
from natgrad.derivatives import laplace
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
parser.add_argument(
    "--N_data",
    help="number of boundary collocation points",
    default=40,
    type=int,
)
args = parser.parse_args()

ITER = args.iter
LM = args.LM
N_Omega = args.N_Omega
N_Gamma = args.N_Gamma
N_data = args.N_data

print(
    f"POISSON INVERSE with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}, "
    f"N_data={N_data}."
)

# random seed for model weigths
seed = 0

alpha = 1.0
beta = 1.0
gamma = 1.0
eta = 0.001

# model_u
activation_u = lambda x: jnp.tanh(x)
layer_sizes_u = [3, 64, 1]
params_u = mlp.init_params(layer_sizes_u, random.PRNGKey(seed))
f_params_u, unravel_u = ravel_pytree(params_u)
model_u = mlp.mlp(activation_u)
v_model_u = vmap(model_u, (None, 0))

# model_f, this is the control variable f
activation_f = lambda x: jnp.tanh(x)
layer_sizes_f = [3, 32, 1]
params_f = mlp.init_params(layer_sizes_f, random.PRNGKey(0))
f_params, unravel_f = ravel_pytree(params_f)
model_f = mlp.mlp(activation_f)
v_model_f = vmap(model_f, (None, 0))

# collocation points
dim = 3
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle([(0.0, 1.0) for _ in range(0, dim)])
boundary = HyperrectangleBoundary(intervals)
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=N_Omega)
x_data = interior.random_integration_points(random.PRNGKey(0), N=N_data)
x_eval = interior.random_integration_points(random.PRNGKey(999), N=10 * N_Omega)
x_Gamma = boundary.random_integration_points(random.PRNGKey(0), N=N_Gamma)


@jit
def u_star(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    return jnp.reshape(
        jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        * jnp.exp(-jnp.pi * jnp.sqrt(2.0) * z),
        (1),
    )


v_u_star = vmap(u_star, (0))


# interior residual
def interior_res(params_u, params_f, x):
    return laplace(model_u, argnum=1)(params_u, x) + model_f(params_f, x)


v_interior_res = vmap(interior_res, (None, None, 0))

# data residual
data_res = lambda params_u, x: model_u(params_u, x) - u_star(x)
v_data_res = vmap(data_res, (None, 0))

# boundary residual
boundary_res = lambda params_u, x: model_u(params_u, x) - u_star(x)
v_boundary_res = vmap(boundary_res, (None, 0))

# regularization residual
regularization_res = lambda params_f, x: model_f(params_f, x)
v_regularization_res = vmap(regularization_res, (None, 0))


# loss function
def loss(params_u, params_f):
    return (
        0.5 * jnp.mean(v_interior_res(params_u, params_f, x_Omega) ** 2)
        + 0.5 * jnp.mean(v_data_res(params_u, x_data) ** 2)
        + 4.0 * 0.5 * jnp.mean(v_boundary_res(params_u, x_Gamma) ** 2)
        + 0.5 * eta * jnp.mean(v_regularization_res(params_f, x_Omega) ** 2)
    )


# define gramians
gram_A_1 = jit(gram_factory(interior_res, argnum_1=0, argnum_2=0))
gram_A_2 = jit(gram_factory(boundary_res, argnum_1=0, argnum_2=0))
gram_A_3 = jit(gram_factory(data_res, argnum_1=0, argnum_2=0))


def gram_A(params_u, params_f, x_Omega, x_Gamma, x_data):
    return (
        gram_A_1(params_u, params_f, x=x_Omega)
        + 4.0 * gram_A_2(params_u, x=x_Gamma)
        + gram_A_3(params_u, x=x_data)
    )


gram_B = jit(gram_factory(interior_res, argnum_1=1, argnum_2=0))
gram_D = jit(gram_factory(interior_res, argnum_1=1, argnum_2=1))
gram_D_reg = jit(gram_factory(regularization_res, argnum_1=0, argnum_2=0))


@jit
def gram(params_u, params_f, x_Omega, x_Gamma, x_data):
    A = gram_A(params_u, params_f, x_Omega, x_Gamma, x_data)
    B = gram_B(params_u, params_f, x=x_Omega)
    C = jnp.transpose(B)
    D = gram_D(params_u, params_f, x=x_Omega) + eta * gram_D_reg(params_f, x=x_Omega)
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

error_f = lambda x: model_f(params_f, x)
v_error_f = vmap(error_f, (0))


def l2_norm(f, x_eval):
    return jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error_u = l2_norm(v_error_u, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)


for iteration in range(ITER):
    grad_u = grad(loss, 0)(params_u, params_f)
    grad_f = grad(loss, 1)(params_u, params_f)

    gram_matrix = gram(params_u, params_f, x_Omega, x_Gamma, x_data)

    # Marquardt-Levenberg
    Id = jnp.identity(len(gram_matrix))
    gram_matrix = jnp.min(jnp.array([loss(params_u, params_f), LM])) * Id + gram_matrix

    flat_combined_grad, retrieve_pytrees = flatten_pytrees(grad_u, grad_f)
    long_flat_nat_grad = lstsq(gram_matrix, flat_combined_grad)[0]
    nat_grad_u, nat_grad_f = retrieve_pytrees(long_flat_nat_grad)

    params_u, params_f, actual_step = ls_update(
        params_u, params_f, nat_grad_u, nat_grad_f
    )

    if iteration % 50 == 0:
        l2_error_u = l2_norm(v_error_u, x_eval)
        h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)
        l2_error_f = l2_norm(v_error_f, x_eval)

        print(
            f"ENGD Iteration: {iteration} with loss: "
            f"{loss(params_u, params_f)} and step: {actual_step} "
            f"L2 u {l2_error_u} H1 u {h1_error_u} L2 f {l2_error_f}"
        )

l2_error_u = l2_norm(v_error_u, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_u_abs_grad, x_eval)
l2_error_f = l2_norm(v_error_f, x_eval)
print(
    f"INVERSE POISSON: loss {loss(params_u, params_f)} "
    f"L2 u {l2_error_u} H1 u {h1_error_u} L2 f {l2_error_f}"
)
