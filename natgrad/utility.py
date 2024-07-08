import jax.numpy as jnp
from jax import jit, vmap


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
