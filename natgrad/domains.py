from typing import Any

import jax
import jax.numpy as jnp
from jax import random


class Hyperrectangle:
    """
    A product of intervals in R^d.

    The hyperrectangle is specified as a product of intervals.
    For example

    intervals = ((0., 1.), (0., 1.), (0., 1.))

    is the unit cube in R^3. The assumption is that intervals
    is convertable to an array of shape (d, 2).

    Note that no method for deterministic integration points is
    provided in this class. The Hyperrectangle is potentially a high
    dimensional object. Deterministic integration points should be
    implemented in child classes.

    Parameters
    ----------
    intervals
        An iterable of intervals, see example above.

    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)

        l_bounds = None
        r_bounds = None

        if jnp.shape(self._intervals) == (2,):
            l_bounds = self._intervals[0]
            r_bounds = self._intervals[1]

        else:
            l_bounds = self._intervals[:, 0]
            r_bounds = self._intervals[:, 1]

        self._l_bounds = jnp.reshape(
            jnp.asarray(l_bounds, dtype=float),
            newshape=(-1),
        )

        self._r_bounds = jnp.reshape(
            jnp.asarray(r_bounds, dtype=float),
            newshape=(-1),
        )

        if len(self._l_bounds) != len(self._r_bounds):
            raise ValueError(
                "[In constructor of Hyperrectangle]: intervals "
                "is not convertable to an array of shape (d, 2)."
            )

        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(
                "[In constructor of Hyperrectangle]: The "
                "lower bounds must be smaller than the upper bounds."
            )

        self._dimension = len(self._l_bounds)

    def measure(self) -> float:
        return jnp.product(self._r_bounds - self._l_bounds)

    def random_integration_points(self, key: Any, N: int = 50):
        """
        N uniformly drawn collocation points in the hyperrectangle.

        Parameters
        ----------
        key
            A random key from jax.random.PRNGKey(<int>).
        N=50: int
            Number of random points.

        """
        return random.uniform(
            key,
            shape=(N, self._dimension),
            minval=jnp.broadcast_to(
                self._l_bounds,
                shape=(N, self._dimension),
            ),
            maxval=jnp.broadcast_to(
                self._r_bounds,
                shape=(N, self._dimension),
            ),
        )

    def distance_function(
        self,
        x,
    ):
        """
        A smooth approximation of the distance fct to the boundary.

        Note that when using this function in implementations for
        loss functions one should explicitly vectorize it using
        for instance vmap(distance_function, (0)) to let it act on
        arrays of shape (n, d) and return (n,).

        Parameters
        ----------
        x: Float[Array, "d"]
            A single spatial point x of shape (d,) where d is the
            dimension of the Hyperrectangle.
        """

        return jnp.product((x - self._l_bounds) * (x - self._r_bounds))


class HypercubeInitial:
    """
    Initial time domain of a unit hypercube. Convention: x_0 = time.

    """

    def __init__(self, dim):
        if not isinstance(dim, int):
            raise TypeError("[Constructor HypercubeInitial:] dim " "must be integer")

        self._dim = dim
        self._hypercube = Hyperrectangle([(0.0, 1.0) for _ in range(0, self._dim)])

    def measure(self):
        return 1.0

    def random_integration_points(self, key, N=50):
        x = self._hypercube.random_integration_points(key, N)

        for i in range(0, N):
            x = x.at[i, 0].set(0.0)

        return x

    def distance_function(self, x):
        pass


def in_interval(xi: float, intervals) -> int:
    """
    xi should be in [0,length] and the function returns the interval
    number xi lies in

    """
    xi = float(xi)
    start = 0.0
    end = 0.0
    for i, len_i in enumerate(intervals[:, 1] - intervals[:, 0]):
        end += float(len_i)
        if xi >= start and xi <= end:
            return i

    return ValueError(
        "[in_interval] Invalid value to determine " "which boundary to project to."
    )


class HypercubeParabolicBoundary:
    """
    Parabolic Boundary of the UNIT HyperCube

    The first dimension is always time.
    """

    def __init__(self, dim):
        if not isinstance(dim, int):
            raise TypeError("[Constructor HypercubeBoundary:] dim " "must be integer")

        self._dim = dim
        self._hypercube = Hyperrectangle([(0.0, 1.0) for _ in range(0, self._dim)])

    def measure(self):
        return 2.0 * self._dim - 1.0

    def random_integration_points(self, key, N=50):
        x = self._hypercube.random_integration_points(key, N)

        for i in range(0, N):
            # advance random key
            key_0, key_1 = jax.random.split(key, num=2)
            key = key_0

            # 0 or 1 depending on side
            rand_side = float(jax.random.randint(key_0, shape=(), minval=0, maxval=2))

            # 0, ..., dim-1, determines which coordinate is set to 0 or 1
            rand_dim = jax.random.randint(key_1, shape=(), minval=0, maxval=self._dim)

            # project to random sides of the Hypercubes boundary
            x = x.at[i, rand_dim].set(rand_side)

            # take out points corresponding to final time
            check_final = []
            for i in range(len(x)):
                if x[i, 0] == 1.0:
                    check_final.append(False)
                else:
                    check_final.append(True)

        return x[jnp.array(check_final)]

    def distance_function(self, x):
        pass


class HyperrectangleParabolicBoundary:
    """
    Boundary of a Hyperrectangle

    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)
        # add check
        self._hyperrectangle = Hyperrectangle(intervals)

    def measure(self):
        return jnp.prod(self._intervals[:, 1] - self._intervals[:, 0])

    def random_integration_points(self, key, N=50):
        x = self._hyperrectangle.random_integration_points(key, N)
        length = jnp.sum(self._intervals[:, 1] - self._intervals[:, 0])

        for i in range(0, N):
            # advance random key
            key_0, key_1 = jax.random.split(key, num=2)
            key = key_0

            # location is float in [0, sum of interval lengths]
            location = random.uniform(key_1, (), minval=0.0, maxval=length)
            rand_dim = in_interval(location, self._intervals)

            # 0 or 1 depending on side
            index = jax.random.randint(key_0, shape=(), minval=0, maxval=2)
            rand_side = self._intervals[rand_dim, index]

            # project to random sides of the Hypercubes boundary
            x = x.at[i, rand_dim].set(rand_side)

        # take out points corresponding to final time
        check_final = []
        for i in range(len(x)):
            if x[i, 0] == 1.0:
                check_final.append(False)
            else:
                check_final.append(True)

        return x[check_final, :]

    def distance_function(self, x):
        pass


class HypercubeBoundary:
    """
    Boundary of the UNIT HyperCube

    """

    def __init__(self, dim):
        if not isinstance(dim, int):
            raise TypeError("[Constructor HypercubeBoundary:] dim " "must be integer")

        self._dim = dim
        self._hypercube = Hyperrectangle([(0.0, 1.0) for _ in range(0, self._dim)])

    def measure(self):
        return 2.0 * self._dim

    def random_integration_points(self, key, N=50):
        x = self._hypercube.random_integration_points(key, N)

        for i in range(0, N):
            # advance random key
            key_0, key_1 = jax.random.split(key, num=2)
            key = key_0

            # 0 or 1 depending on side
            rand_side = float(jax.random.randint(key_0, shape=(), minval=0, maxval=2))

            # 0, ..., dim-1, determines which coordinate is set to 0 or 1
            rand_dim = jax.random.randint(key_1, shape=(), minval=0, maxval=self._dim)

            # project to random sides of the Hypercubes boundary
            x = x.at[i, rand_dim].set(rand_side)

        return x

    def distance_function(self, x):
        pass
