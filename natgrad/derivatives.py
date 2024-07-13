"""
Contains implementations of differential operators.

"""

import jax.numpy as jnp
from jax import hessian, jacrev


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


def del_i(
    func,
    argnum: int = 0,
):
    """
    Partial derivative for a function of signature (d,) ---> ().

    """

    def func_splitvar(*args):
        return func(jnp.array(args))

    d_splitvar_di = jacrev(func_splitvar, argnum)

    def dfunc_di(x):
        return d_splitvar_di(*x)

    return dfunc_di


def div(func, argnum=0):
    def div_f(*args, **kwargs):
        J = jacrev(func, argnum)(*args, **kwargs)
        return jnp.trace(J, axis1=-2, axis2=-1)

    return div_f


# should be applied to functions (d,) -> (d,)
def symgrad(func, argnum=0):
    def eps_u(*args, **kwargs):
        Du = jacrev(func, argnum)(*args, **kwargs)
        return 0.5 * (Du + jnp.transpose(Du))

    return eps_u
