"""
Mixed Laplace

The concrete example is:

min E(u,v) = ||nabla(u) - v||^2 + ||f + div(v)||^2 + ||u - g||^2_Gamma

the solution is
u(x,y,z) = sin(pi x)sin(pi y)sin(pi z)
v(x,y,z) = (pi cos(pi x)sin(pi y)sin(pi z), sin(pi x)pi cos(pi y)sin(pi z), sin(pi x)sin(pi y)pi cos(pi z))

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
from natgrad.derivatives import div
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

print(f"DARCY with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}")

# random seed for model weigths
seed = 0

# model_u
activation_u = lambda x: jnp.tanh(x)
layer_sizes_u = [3, 32, 1]
params_u = mlp.init_params(layer_sizes_u, random.PRNGKey(seed))
f_params_u, unravel_u = ravel_pytree(params_u)
model_u = mlp.mlp(activation_u)
v_model_u = vmap(model_u, (None, 0))

# model_f, this is the control variable f
activation_v = lambda x: jnp.tanh(x)
layer_sizes_v = [3, 32, 3]
params_v = mlp.init_params(layer_sizes_v, random.PRNGKey(0))
v_params, unravel_v = ravel_pytree(params_v)
model_v = mlp.mlp(activation_v)
v_model_v = vmap(model_v, (None, 0))

# collocation points
dim = 3
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle([(0.0, 1.0) for _ in range(0, dim)])
boundary = HyperrectangleBoundary(intervals)
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=N_Omega)
x_eval = interior.random_integration_points(random.PRNGKey(999), N=10 * N_Omega)
x_Gamma = boundary.random_integration_points(random.PRNGKey(0), N=N_Gamma)


# solution and right-hand side
def u_star(x):
    return jnp.prod(jnp.sin(jnp.pi * x), keepdims=True)


v_u_star = vmap(u_star, (0))


def f(x):
    return 3.0 * jnp.pi**2 * u_star(x)


@jit
def v_star(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    v_1 = jnp.pi * jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
    v_2 = jnp.pi * jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * z)
    v_3 = jnp.pi * jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.cos(jnp.pi * z)
    return jnp.array([v_1, v_2, v_3])


# interior residuals
def mixed_res(params_u, params_v, x):
    grad_u = jacrev(model_u, argnums=1)(params_u, x)
    return grad_u - model_v(params_v, x)


v_mixed_res = vmap(mixed_res, (None, None, 0))


def div_res(params_v, x):
    div_v = div(model_v, argnum=1)(params_v, x)
    return f(x) + div_v


v_div_res = vmap(div_res, (None, 0))

# boundary residual
boundary_res = lambda params_u, x: model_u(params_u, x) - u_star(x)
v_boundary_res = vmap(boundary_res, (None, 0))


# loss function
def loss(params_u, params_v):
    return (
        0.5 * jnp.mean(v_mixed_res(params_u, params_v, x_Omega) ** 2)
        + 0.5 * jnp.mean(v_div_res(params_v, x_Omega) ** 2)
        + 4.0 * 0.5 * jnp.mean(v_boundary_res(params_u, x_Gamma) ** 2)
    )


# define gramians
gram_A_1 = jit(gram_factory(mixed_res, argnum_1=0, argnum_2=0))
gram_A_2 = jit(gram_factory(boundary_res))


def gram_A(params_u, params_v, x_Omega, x_Gamma):
    return gram_A_1(params_u, params_v, x=x_Omega) + 4.0 * gram_A_2(params_u, x=x_Gamma)


gram_B = jit(gram_factory(mixed_res, argnum_1=1, argnum_2=0))

gram_D_1 = jit(gram_factory(mixed_res, argnum_1=1, argnum_2=1))
gram_D_2 = jit(gram_factory(div_res))


def gram_D(params_u, params_v, x_Omega):
    return gram_D_1(params_u, params_v, x=x_Omega) + gram_D_2(params_v, x=x_Omega)


@jit
def gram(params_u, params_v, x_Omega, x_Gamma):
    A = gram_A(params_u, params_v, x_Omega, x_Gamma)
    B = gram_B(params_u, params_v, x=x_Omega)
    C = jnp.transpose(B)
    D = gram_D(params_u, params_v, x_Omega)
    col_1 = jnp.concatenate((A, C), axis=0)
    col_2 = jnp.concatenate((B, D), axis=0)
    return jnp.concatenate((col_1, col_2), axis=1)


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error_u = lambda x: jnp.reshape(model_u(params_u, x) - u_star(x), ())
error_v = lambda x: model_v(params_v, x) - v_star(x)
v_error_u = vmap(error_u, (0))
v_error_v = vmap(error_v, (0))
v_error_abs_grad_u = vmap(lambda x: jnp.sum(jacrev(error_u)(x) ** 2.0) ** 0.5)

error_divv = vmap(lambda x: jnp.trace(jacrev(error_v)(x)))


def l2_norm(f, x_eval):
    return jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error_u = l2_norm(v_error_u, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_abs_grad_u, x_eval)


l2_error_v = l2_norm(v_error_v, x_eval)
hdiv_error_v = l2_error_v + l2_norm(error_divv, x_eval)

for iteration in range(ITER):
    grad_u = grad(loss, 0)(params_u, params_v)
    grad_v = grad(loss, 1)(params_u, params_v)

    gram_matrix = gram(params_u, params_v, x_Omega, x_Gamma)

    # Marquardt-Levenberg

    Id = jnp.identity(len(gram_matrix))
    gram_matrix = jnp.min(jnp.array([loss(params_u, params_v), LM])) * Id + gram_matrix

    flat_combined_grad, retrieve_pytrees = flatten_pytrees(grad_u, grad_v)
    long_flat_nat_grad = lstsq(gram_matrix, flat_combined_grad)[0]
    nat_grad_u, nat_grad_v = retrieve_pytrees(long_flat_nat_grad)

    params_u, params_v, actual_step = ls_update(
        params_u, params_v, nat_grad_u, nat_grad_v
    )

    if iteration % 50 == 0:
        # errors
        l2_error_u = l2_norm(v_error_u, x_eval)
        l2_error_v = l2_norm(v_error_v, x_eval)
        h1_error_u = l2_error_u + l2_norm(v_error_abs_grad_u, x_eval)

        l2_error_v = l2_norm(v_error_v, x_eval)
        hdiv_error_v = l2_error_v + l2_norm(error_divv, x_eval)

        print(
            f"ENGD Iteration: {iteration} with loss: "
            f"{loss(params_u, params_v)} and step: {actual_step} "
            f"L2: u {l2_error_u} and error H1 u :{h1_error_u} "
            f"L2: v {l2_error_v} and error Hdiv v :{hdiv_error_v} "
            f"and step {actual_step}"
        )

l2_error_u = l2_norm(v_error_u, x_eval)
l2_error_v = l2_norm(v_error_v, x_eval)
h1_error_u = l2_error_u + l2_norm(v_error_abs_grad_u, x_eval)
hdiv_error_v = l2_error_v + l2_norm(error_divv, x_eval)
print(
    f"DARCY with loss: "
    f"{loss(params_u, params_v)} "
    f"L2: u {l2_error_u} and error H1 u :{h1_error_u} "
    f"L2: v {l2_error_v} and error Hdiv v :{hdiv_error_v} "
    f"and step {actual_step}"
)
