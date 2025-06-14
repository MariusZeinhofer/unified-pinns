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
    f"OPTIMAL CONTROL with ITER={ITER}, LM={LM}, N_Omega={N_Omega}, N_Gamma={N_Gamma}, "
    f"SEED={seed}."
)

# Initialize training components
key = jax.random.PRNGKey(seed)
sampler_key, eval_key, y_key, p_key, key = jax.random.split(key, num=5)

# distance function to boundary
dist_fct = lambda x: x[0] * (1 - x[0]) * x[1] * (1 - x[1])

# model for state y
y_activation = lambda x: jnp.tanh(x)
y_layer_sizes = [2, 16, 1]
y_params = mlp.init_params(y_layer_sizes, y_key)
_y_model = mlp.mlp(y_activation)
y_model = lambda params, x: _y_model(params, x) * dist_fct(x)
f_y_params, y_unravel = ravel_pytree(y_params)

# model for control p
p_activation = lambda x: jnp.tanh(x)
p_layer_sizes = [2, 8, 1]
p_params = mlp.init_params(p_layer_sizes, p_key)
_p_model = mlp.mlp(p_activation)
p_model = lambda params, x: _p_model(params, x) * dist_fct(x)
f_p_params, p_unravel = ravel_pytree(p_params)

# put params together, unclear if needed...
params = (y_params, p_params)
f_params, unravel = ravel_pytree(params)

# collocation points
dim = 2
intervals = [(0.0, 1.0) for _ in range(0, dim)]
interior = Hyperrectangle(intervals)
x_Omega = interior.random_integration_points(sampler_key, N=N_Omega)
x_eval = interior.random_integration_points(eval_key, N=10 * N_Omega)

# PDE data and manufactured solutions
alpha = 0.1
y_star = lambda x: jnp.prod(jnp.sin(jnp.pi * x), keepdims=True)
p_star = lambda x: x[0] * (1 - x[0]) * x[1] * (1 - x[1])
u_star = lambda x: (1.0 / alpha) * p_star(x)
f = lambda x: dim * jnp.pi**2 * jnp.prod(jnp.sin(jnp.pi * x)) - u_star(x)
y_data = lambda x: -2 * (x[0] * (1 - x[0]) + x[1] * (1 - x[1])) + y_star(x)


# define ingredients for loss functions
def residual_y(y_params, p_params, x):
    lap_y = laplace(y_model, argnum=1)(y_params, x)
    return lap_y + f(x) + (1.0 / alpha) * p_model(p_params, x)


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

    return jax.vmap(f_fct_p, (None, 0))(p_params, X)


@jax.jit
def assemble_gramian(y_params, p_params, X):
    J = assemble_J(y_params, X)
    H = assemble_H(y_params, X)
    A = 1.0 / len(X) * (J.T @ J + H.T @ H)

    J_bar = assemble_J_bar(p_params, X)
    H_bar = assemble_H_bar(p_params, X)
    C = 1.0 / len(X) * (J_bar.T @ J_bar + 1.0 / (alpha**2) * H_bar.T @ H_bar)

    B = 1.0 / (len(X) * alpha) * J.T @ H_bar + 1.0 / len(X) * H.T @ J_bar

    # concat code from ChatGPT
    top = jnp.concatenate([A, B], axis=1)
    bottom = jnp.concatenate([B.T, C], axis=1)

    return jnp.concatenate([top, bottom], axis=0)


# error metrics
def l2_error_y(y_params, X):
    return (
        jnp.mean(jax.vmap(lambda x: (y_model(y_params, x) - y_star(x)) ** 2)(X)) ** 0.5
    )


def l2_error_p(p_params, X):
    return (
        jnp.mean(jax.vmap(lambda x: (p_model(p_params, x) - p_star(x)) ** 2)(X)) ** 0.5
    )


lr = 1e-3

for iteration in range(100_000):
    if method == "GD":
        # autodiff magic
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(
            y_params, p_params, x_Omega
        )

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
        loss, grads = jax.value_and_grad(loss_fct, argnums=(0, 1))(
            y_params, p_params, x_Omega
        )
        f_grads = ravel_pytree(grads)[0]

        # build and regularize the Gramian
        G = assemble_gramian(y_params, p_params, x_Omega)
        G += 1e-6 * jnp.identity(len(G))

        # compute natural gradient
        f_nat_grad = lstsq(G, f_grads, rcond=-1)[0]
        nat_grads = unravel(f_nat_grad)

        # param update
        lr = 1e-2
        params = jax.tree.map(lambda K, dK: K - lr * dK, params, nat_grads)
        y_params, p_params = params

        if iteration % 100 == 0:
            print(
                f"Iter {iteration}, loss {loss}, y_error {l2_error_y(y_params, x_eval)}, "
                f"p_error {l2_error_p(p_params, x_eval)}"
            )


if __name__ == "__main__":
    x = jnp.array([1.0, 0.0])
    X = jnp.array([[1.0, 0.0], [1.0, 0.5]])

    print(
        f"y_model(y_params, x)={y_model(y_params, x)}",
        "shape",
        y_model(y_params, x).shape,
    )  # of shape (1,)
    print(f"y_star(x)={y_star(x)}", "shape", y_star(x).shape)  # of shape (1,)

    print(
        f"lap_y shape: {laplace(y_model, argnum=1)(y_params, x).shape}"
    )  # of shape (1,)
    print(
        f"shape residual_y output {v_residual_y(y_params, p_params, X).shape}"
    )  # of shape (2, 1)

    print(f"loss_fct value {loss_fct(y_params, p_params, x_Omega)}")  # of shape ()
