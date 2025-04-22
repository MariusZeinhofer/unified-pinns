"""L2 penalty approach to the optimality system."""

import argparse
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.numpy.linalg import lstsq
from matplotlib import pyplot as plt

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

# from cartesian to polar
def cart2pol(xy):
    x = xy[0]
    y = xy[1]
    r = jnp.sqrt(x**2 + y**2)

    phi = jnp.arctan2(y, x)
    phi = jnp.where(phi< 0, phi + 2*jnp.pi, phi)
    return jnp.array([r, phi])

v_cart2pol = jax.vmap(cart2pol)

# from polar to cartesian
def pol2cart(rphi):
    r = rphi[0]
    phi = rphi[1]

    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    return jnp.array([x, y])

v_pol2cart = jax.vmap(pol2cart)

print(
    f"OPTIMAL CONTROL with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}, "
    f"SEED={seed}, METHOD={method}."
)

# Initialize training components
key = jax.random.PRNGKey(seed)
sampler_key, eval_key, y_key, p_key, key = jax.random.split(key, num=5)

def dist_fct(x):
    """Zero on boundary of annulus of radii 1, 3."""
    r, _ = cart2pol(x)
    return (r - 1) * (3 - r)

def bc(x):
    """The dirichlet bc of y"""
    r, _ = cart2pol(x)
    return r ** 2

# model for state y
y_activation = lambda x: jnp.tanh(x)
y_layer_sizes = [2, 32, 1]
y_params = mlp.init_params(y_layer_sizes, y_key)
_y_model = mlp.mlp(y_activation)
y_model = lambda params, x : _y_model(params, x) * dist_fct(x) + bc(x)
f_y_params, y_unravel = ravel_pytree(y_params)

# model for control p
p_activation = lambda x: jnp.tanh(x)
p_layer_sizes = [2, 16, 1]
p_params = mlp.init_params(p_layer_sizes, p_key)
_p_model = mlp.mlp(p_activation)
p_model = lambda params, x : _p_model(params, x) * dist_fct(x)
f_p_params, p_unravel = ravel_pytree(p_params)

# put params together
params = (y_params, p_params)
f_params, unravel = ravel_pytree(params)

# collocation points
dim = 2
intervals = [(-3.0, 3.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
_x_Omega = interior.random_integration_points(sampler_key, N=20_000)

# we need to guarantee that the points lie in the annulus.
# this is hacky, needs to be done correctly
pol_x_Omega = v_cart2pol(_x_Omega)
radii = pol_x_Omega[:, 0]
mask = (radii >= 1.0) & (radii <= 3.0)
filtered_pol_x_Omega = pol_x_Omega[mask]
X = v_pol2cart(filtered_pol_x_Omega)

x_Omega = X[0:1000]
x_eval = X[1000:]

# the pointwise projection, acts on arrays of arbitrary shape
def projection(x, lower=-0.5, upper=0.7):
    return jnp.minimum(jnp.maximum(x, lower), upper)

def projection_prime(x, lower=-0.5, upper=0.7):
    return jnp.where((x > lower) & (x < upper), 1.0, 0.0)

# PDE data and manufactured solutions
alpha = 0.01
y_star = lambda x: cart2pol(x)[0] ** 2

def p_star(x):
    r, theta = cart2pol(x)
    return alpha * (r - 1) * (r - 3) * jnp.cos(theta)

def u_star(x):
    r, theta = cart2pol(x)
    return projection(-(r - 1) * (r - 3) * jnp.cos(theta))

f = lambda x: -4 - u_star(x)

def y_data(x):
    r, theta = cart2pol(x)
    return r ** 2 + 3 * (1 - 1 / r ** 2) * alpha * jnp.cos(theta)

# define ingredients for loss functions
def residual_y(y_params, p_params, x):
    lap_y = laplace(y_model, argnum=1)(y_params, x)
    return lap_y + f(x) + projection(-(1. / alpha) * p_model(p_params, x))

v_residual_y = jax.vmap(residual_y, (None, None, 0))

def residual_p(y_params, p_params, x):
    lap_p = laplace(p_model, argnum=1)(p_params, x)
    return lap_p + y_model(y_params, x) - y_data(x)

v_residual_p = jax.vmap(residual_p, (None, None, 0))

@jax.jit
def loss_fct(y_params, p_params, X):
    loss_y = 0.5 * jnp.mean(v_residual_y(y_params, p_params, X) ** 2)
    loss_p = 0.5 * jnp.mean(v_residual_p(y_params, p_params, X) ** 2)
    return loss_y + loss_p

# Gauss-Newton matrix builders
@jax.jit
def assemble_J(y_params, X):
    def f_grad_lap_y(y_params, x):
        lap_y = lambda y_params, x: laplace(y_model, argnum=1)(y_params, x).squeeze()
        return ravel_pytree(jax.grad(lap_y)(y_params, x))[0]
    return jax.vmap(f_grad_lap_y, (None, 0))(y_params, X)

@jax.jit
def assemble_H(y_params, X):
    def f_fct_y(y_params, x):
        fct_y = lambda y_params, x: y_model(y_params, x).squeeze()
        return ravel_pytree(jax.grad(fct_y)(y_params, x))[0]
    return jax.vmap(f_fct_y, (None, 0))(y_params, X)

@jax.jit
def assemble_J_bar(p_params, X):
    def f_grad_lap_p(p_params, x):
        lap_p = lambda p_params, x: laplace(p_model, argnum=1)(p_params, x).squeeze()
        return ravel_pytree(jax.grad(lap_p)(p_params, x))[0]
    return jax.vmap(f_grad_lap_p, (None, 0))(p_params, X)

@jax.jit
def assemble_H_bar(p_params, X):
    def f_fct_p(p_params, x):
        fct_p = lambda p_params, x: p_model(p_params, x).squeeze()
        return ravel_pytree(jax.grad(fct_p)(p_params, x))[0]
    
    #P = projection_prime(-1. / alpha * jax.vmap(p_model, (None, 0))(p_params, X))
    P = -1. / alpha * projection_prime(-1. / alpha * jax.vmap(p_model, (None, 0))(p_params, X))

    return P * jax.vmap(f_fct_p, (None, 0))(p_params, X)


@jax.jit
def assemble_gramian(y_params, p_params, X):
    J = assemble_J(y_params, X)
    H = assemble_H(y_params, X)
    A = 1. / len(X) * (J.T @ J + H.T @ H)
    
    J_bar = assemble_J_bar(p_params, X)
    H_bar = assemble_H_bar(p_params, X)
    C = 1. / len(X) * (J_bar.T @ J_bar + H_bar.T @ H_bar)

    B = 1. / (len(X)) * J.T @ H_bar + 1. / len(X) * H.T @ J_bar

    # concat code from ChatGPT
    top = jnp.concatenate([A, B], axis=1)
    bottom = jnp.concatenate([B.T, C], axis=1)
    
    return jnp.concatenate([top, bottom], axis=0)

# error metrics
def l2_error_y(y_params, X):
    return jnp.mean(jax.vmap(lambda x: (y_model(y_params, x) - y_star(x)) ** 2)(X)) ** 0.5

def l2_error_p(p_params, X):
    return jnp.mean(jax.vmap(lambda x: (p_model(p_params, x) - p_star(x)) ** 2)(X)) ** 0.5

lr = 1e-2

for iteration in range(100000):

    if method == "GD":
        # autodiff magic
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(y_params, p_params, x_Omega)

        # param update
        params = jax.tree.map(lambda K, dK: K - lr * dK, params, grads)

        y_params, p_params = params

        if iteration % 100 == 0:
            print(
                f"Iter {iteration}, loss {loss}, y_error {l2_error_y(y_params, x_eval)}, "
                f"p_error {l2_error_p(p_params, x_eval)}"
            )

    if method == "GN":
        # autodiff magic
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(y_params, p_params, x_Omega)
        f_grads = ravel_pytree(grads)[0]

        # build and regularize the Gramian
        G = assemble_gramian(y_params, p_params, x_Omega)
        G += 1e-5 * jnp.identity(len(G))

        # compute natural gradient
        f_nat_grad = lstsq(G, f_grads, rcond=-1)[0]
        nat_grads = unravel(f_nat_grad)

        # param update
        params = jax.tree.map(lambda K, dK: K - lr * dK, params, nat_grads)
        y_params, p_params = params

        
        if iteration % 100 == 0:
            print(
                f"Iter {iteration}, loss {loss}, y_error {l2_error_y(y_params, x_eval)}, "
                f"p_error {l2_error_p(p_params, x_eval)}"
            )