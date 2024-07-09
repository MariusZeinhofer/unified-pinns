import jax.numpy as jnp
from jax import jit, vmap
from jax.flatten_util import ravel_pytree


def grid_line_search_factory(loss, steps):
    def loss_at_step(step, params, tangent_params):
        updated_params = [
            (w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, tangent_params)
        ]
        return loss(updated_params)

    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None)))

    @jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        return [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, tangent_params)
        ], step_size

    return grid_line_search_update


def flatten_pytrees(pytree_1, pytree_2):
    f_pytree_1, unravel_1 = ravel_pytree(pytree_1)
    f_pytree_2, unravel_2 = ravel_pytree(pytree_2)

    len_1 = len(f_pytree_1)
    len_2 = len(f_pytree_2)
    flat = jnp.concatenate([f_pytree_1, f_pytree_2], axis=0)

    def retrieve_pytrees(flat):
        flat_1 = flat[0:len_1]
        flat_2 = flat[len_1 : len_1 + len_2]
        return unravel_1(flat_1), unravel_2(flat_2)

    return flat, retrieve_pytrees


def two_variable_grid_line_search_factory(loss, steps):
    def loss_at_step(
        step,
        params_u,
        params_v,
        tangent_params_u,
        tangent_params_v,
    ):
        updated_params_u = [
            (w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params_u, tangent_params_u)
        ]
        updated_params_v = [
            (w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params_v, tangent_params_v)
        ]
        return loss(updated_params_u, updated_params_v)

    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None, None, None)))

    @jit
    def grid_line_search_update(
        params_u,
        params_v,
        tangent_params_u,
        tangent_params_v,
    ):
        losses = v_loss_at_steps(
            steps, params_u, params_v, tangent_params_u, tangent_params_v
        )
        step_size = steps[jnp.argmin(losses)]

        new_params_u = [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params_u, tangent_params_u)
        ]

        new_params_v = [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params_v, tangent_params_v)
        ]

        return new_params_u, new_params_v, step_size

    return grid_line_search_update
