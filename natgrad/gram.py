import jax.numpy as jnp
from jax import jacfwd, jacrev, vmap
from jax.flatten_util import ravel_pytree


def flat(unravel, argnum=0):
    def flatten(func):
        def flattened(*args, **kwargs):
            args = [arg for arg in args]
            args[argnum] = unravel(args[argnum])
            return func(*args, **kwargs)

        return flattened

    return flatten


def gram_factory(
    residual,
    argnum_1=0,
    argnum_2=0,
):
    """
    ...

    Parameters
    ----------
    residual: Callable
        Of signature (PyTree, (d_in,)) -> (d_res,) where PyTree may or
        may not be flattened.

    Todos
    -----
        Remove the copies of the input params.

    """

    def v_residual(*params, x):
        nones = [None for _ in params]
        return jnp.reshape(
            vmap(residual, (*nones, 0))(*params, x),
            (-1,),
        )

    def gramian(*params, x):
        """
        ...

        Parameters
        ----------
        *params:
            one or more PyTrees.

        x:
            Array of shape (N, d_in)

        """
        # flatten params to enable correct jacobian compuations
        f_params_1, unravel_1 = ravel_pytree(params[argnum_1])
        f_params_2, unravel_2 = ravel_pytree(params[argnum_2])

        # Determine autodiff for argnum_1
        if len(params[argnum_1]) > len(x):
            jac = jacrev
        else:
            jac = jacfwd

        # Compute the first jacobian
        jac_1 = jac(flat(unravel_1, argnum_1)(v_residual), argnum_1)
        par_1 = [param for param in params]  # copy!
        par_1[argnum_1] = f_params_1
        J_1 = jac_1(*par_1, x=x)

        # If avoidable, don't recompute
        if argnum_1 == argnum_2:
            J_2 = J_1

        else:
            # Determine autodiff for argnum_2
            if len(params[argnum_2]) > len(x):
                jac = jacrev
            else:
                jac = jacfwd

            # Compute second jacobian
            jac_2 = jac(flat(unravel_2, argnum_2)(v_residual), argnum_2)
            par_2 = [param for param in params]  # copy!
            par_2[argnum_2] = f_params_2
            J_2 = jac_2(*par_2, x=x)

        return 1.0 / len(x) * jnp.transpose(J_2) @ J_1

    return gramian
