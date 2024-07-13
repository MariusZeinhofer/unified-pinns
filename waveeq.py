"""
Implementation of 3d wave equation.

Manufactured solution is

u(t,x,y,z) = sin(pi t)sin(pi x)sin(pi y)sin(pi z)

"""

import argparse
import jax
import jax.numpy as jnp
from jax import random
from jax import vmap, jit, grad, jacrev
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq

from natgrad.domains import (
    Hyperrectangle,
    HyperrectangleInitial,
    HyperrectangleParabolicBoundary,
)
import natgrad.mlp as mlp
from natgrad.derivatives import laplace
from natgrad.gram import gram_factory
from natgrad.utility import grid_line_search_factory


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
    "--N_init",
    help="number of boundary collocation points",
    default=100,
    type=int,
)
args = parser.parse_args()

ITER = args.iter
LM = args.LM
N_Omega = args.N_Omega
N_Gamma = args.N_Gamma
N_init = args.N_init

print(
    f"WAVE EQUATION with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}, "
    f"N_init={N_init}"
)

# random seed for model weigths
seed = 0

# model
activation = lambda x: jnp.tanh(x)
layer_sizes = [4, 64, 1]
params = mlp.init_params(layer_sizes, random.PRNGKey(seed))
model = mlp.mlp(activation)
f_params, unravel = ravel_pytree(params)

# collocation points
dim = 4
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
boundary = HyperrectangleParabolicBoundary(intervals)
initial = HyperrectangleInitial(intervals)
x_Omega = interior.random_integration_points(random.PRNGKey(0), N=N_Omega)
x_eval = interior.random_integration_points(random.PRNGKey(999), N=10 * N_Omega)
x_Gamma = boundary.random_integration_points(random.PRNGKey(0), N=N_Gamma)
x_init = initial.random_integration_points(random.PRNGKey(0), N=N_init)


@jit
def u_star(txyz):
    t = txyz[0]
    x = txyz[1]
    y = txyz[2]
    z = txyz[3]
    u = (
        jnp.sin(jnp.pi * t)
        * jnp.sin(jnp.pi * x)
        * jnp.sin(jnp.pi * y)
        * jnp.sin(jnp.pi * z)
    )
    return jnp.reshape(u, (1,))


v_u_star = vmap(u_star, (0))


def f(txyz):
    t = txyz[0:1]
    xyz = txyz[1:]
    L = laplace(lambda xi: u_star(jnp.concatenate([t, xi])))(xyz)
    dtt_u = laplace(lambda s: u_star(jnp.concatenate([s, xyz])))(t)
    return dtt_u - L


# stokes operator residual
def interior_res(params, txyz):
    t = txyz[0:1]
    xyz = txyz[1:]
    dtt_u = laplace(lambda s: model(params, jnp.concatenate([s, xyz])))(t)
    L = laplace(lambda xi: model(params, jnp.concatenate([t, xi])))(xyz)
    return dtt_u - L - f(txyz)


v_interior_res = vmap(interior_res, (None, 0))


# boundary residual
def boundary_res(params, txyz):
    return model(params, txyz) - u_star(txyz)


v_boundary_res = vmap(boundary_res, (None, 0))


def initial_res(params, txyz):
    t = txyz[0:1]
    xyz = txyz[1:]
    dt_u = jacrev(lambda s: model(params, jnp.concatenate([s, xyz])))(t)
    dt_u_star = jacrev(lambda s: u_star(jnp.concatenate([s, xyz])))(t)
    return dt_u - dt_u_star


v_initial_res = vmap(initial_res, (None, 0))


# loss function
def loss(params):
    return (
        0.5 * jnp.mean(v_interior_res(params, x_Omega) ** 2)
        + 0.5 * jnp.mean(v_initial_res(params, x_init) ** 2)
        + 7.0 * 0.5 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)
    )


# gramians
gram_int = jit(gram_factory(interior_res))
gram_init = jit(gram_factory(initial_res))
gram_bdry = jit(gram_factory(boundary_res))

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)

# errors
error = lambda x: jnp.reshape(model(params, x) - u_star(x), ())
v_error = vmap(error, (0))


def spatial_derivative_error(txyz):
    t = txyz[0:1]
    xyz = txyz[1:]
    return jacrev(lambda xi: error(jnp.concatenate([t, xi])))(xyz)


v_error_abs_grad = vmap(lambda x: jnp.sum(spatial_derivative_error(x) ** 2.0) ** 0.5)


def l2_norm(f, x_eval):
    return jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error = l2_norm(v_error, x_eval)
h1_error = l2_error + l2_norm(v_error_abs_grad, x_eval)

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
    G_init = gram_init(params, x=x_init)
    G_bdry = 7.0 * gram_bdry(params, x=x_Gamma)
    G = G_int + G_bdry + G_init

    # Marquardt-Levenberg
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
print(f"WAVE EQUATION: loss {loss(params)} L2 {l2_error} H1 {h1_error}")
