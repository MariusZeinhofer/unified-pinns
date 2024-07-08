"""
Contains implementations of differential operators.

"""

import jax.numpy as jnp
from jax import hessian


def laplace(func, argnum=0):
    """
    Computes laplacian of func with respect to the argument argnum.

    Parameters
    ----------
    func: Callable
        Function whose laplacian should be computed.

    argnum: int
        Argument number wrt which laplacian should be computed.

    Returns
    -------
    Callable of same signature as func.

    Issues
    ------
    Vector valued func. So far not tested if this function works
    appropriately for vector valued functions. We need an
    implementation that does this.

    """
    hesse = hessian(func, argnum)
    return lambda *args, **kwargs: jnp.trace(
        hesse(*args, **kwargs),
        axis1=-2,
        axis2=-1,
    )
