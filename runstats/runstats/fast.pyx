"""Python RunStats

Compute Statistics and Regression in a single pass.

"""

from __future__ import division

from .core import make_statistics, make_regression


cdef class Statistics(object):
    """Compute statistics in a single pass.

    Computes the minimum, maximum, mean, variance, standard deviation,
    skewness, and kurtosis.
    Statistics objects may also be added together and copied.

    Based entirely on the C++ code by John D Cook at
    http://www.johndcook.com/skewness_kurtosis.html
    """
    cdef public double _count
    cdef public double _eta
    cdef public double _rho
    cdef public double _rho2
    cdef public double _tau
    cdef public double _phi
    cdef public double _min
    cdef public double _max
    cdef public double _last
    cdef public double _square
    cdef public double _max_offset

    def __init__(self, iterable=()):
        """Initialize Statistics object.

        Iterates optional parameter `iterable` and pushes each value into the
        statistics summary.
        """
        self.clear()

        for value in iterable:
            self.push(value)

    cpdef clear(self):
        """Clear Statistics object."""
        self._count = self._last = self._eta = self._rho = self._rho2 = self._max_offset = self._tau = self._square = self._phi = 0.0
        self._min = self._max = float('nan')

    def __richcmp__(self, other, op):
        if op == 2:
            return self.get_state() == other.get_state()
        elif op == 3:
            return self.get_state() != other.get_state()
        else:
            return NotImplemented

    def get_state(self):
        """Get internal state."""
        return (
            self._count,
            self._eta,
            self._rho,
            self._rho2,
            self._tau,
            self._phi,
            self._min,
            self._max,
            self._last,
            self._square,
            self._max_offset
        )

    def set_state(self, state):
        """Set internal state."""
        (
            self._count,
            self._eta,
            self._rho,
            self._rho2,
            self._tau,
            self._phi,
            self._min,
            self._max,
            self._last,
            self._square,
            self._max_offset
        ) = state

    @classmethod
    def fromstate(cls, state):
        """Return Statistics object from state."""
        stats = cls()
        stats.set_state(state)
        return stats

    def __reduce__(self):
        return make_statistics, (self.get_state(),)

    def copy(self, _=None):
        """Copy Statistics object."""
        return self.fromstate(self.get_state())

    __copy__ = copy
    __deepcopy__ = copy

    def __len__(self):
        """Number of values that have been pushed."""
        return int(self._count)

    cpdef push(self, double value, cur_max=None):
        """Add `value` to the Statistics summary."""
        cdef double val = value

        if self._count == 0.0:
            self._min = val
            self._max = val
        else:
            self._min = min(self._min, val)
            self._max = max(self._max, val)

        delta = val - self._eta
        delta_n = delta / (self._count + 1)
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * self._count

        self._count += 1
        self._eta += delta_n
        self._phi += (
            term * delta_n2 * (self._count ** 2 - 3 * self._count + 3)
            + 6 * delta_n2 * self._rho
            - 4 * delta_n * self._tau
        )
        self._tau += (
            term * delta_n * (self._count - 2)
            - 3 * delta_n * self._rho
        )
        self._rho += term

        #additions
        if cur_max is not None:
            self._max_offset += (value - cur_max)**2

        self._rho2 += (value - self._last)**2
        self._square += val**2
        self._last = value

    cpdef minimum(self):
        """Minimum of values."""
        return self._min

    cpdef pos_minimum(self, value):
        cdef double val = value
        return min(val,self._min)

    cpdef maximum(self):
        """Maximum of values."""
        return self._max

    cpdef pos_maximum(self, value):
        """Maximum of values."""
        cdef double val = value
        return max(val, self._max)

    cpdef pos_mean(self, value):
        """Mean of values."""
        cdef double val = value
        delta = val - self._eta
        delta_n = delta / (self._count + 1)
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * self._count
        return self._eta + delta_n
        
    cpdef mean(self):
        """Mean of values."""
        return self._eta

    cpdef variance(self, ddof=1.0):
        """Variance of values (with `ddof` degrees of freedom)."""
        if self._count < 2:
            return (self._last - 0.0)**2 / 2
        return self._rho / (self._count - ddof)

    cpdef pos_variance(self, value, ddof=1.0):
        cdef double val = value
        if self._count == 0:
            return (val - 0.0)**2 / 2
        delta = val - self._eta
        delta_n = delta / (self._count + 1)
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * self._count
        return (self._rho + term) / (self._count + 1 - ddof)

    cpdef local_variance(self, ddof=0.0):
        """Variance of values (with `ddof` degrees of freedom)."""
        return self._rho2 / (self._count - ddof)

    cpdef pos_local_variance(self, value, ddof=0.0):
        """Variance of values (with `ddof` degrees of freedom)."""
        cdef double val = value
        return (self._rho2 + (value - self._last)**2)/ (self._count + 1 - ddof)

    cpdef max_offset(self):
        """Variance of values (with `ddof` degrees of freedom)."""
        return self._max_offset 

    cpdef pos_max_offset(self, value, max_):
        """Maximum of values."""
        cdef double val = value
        cdef double _max = max_
        return self._max_offset + (val - _max)**2

    cpdef squares(self):
        return self._square

    cpdef pos_squares(self, value):
        cdef double val = value
        return self._square + val**2

    cpdef stddev(self, ddof=1.0):
        """Standard deviation of values (with `ddof` degrees of freedom)."""
        return self.variance(ddof) ** 0.5

    cpdef skewness(self):
        """Skewness of values."""
        return (self._count ** 0.5) * self._tau / pow(self._rho, 1.5)

    cpdef kurtosis(self):
        """Kurtosis of values."""
        return self._count * self._phi / (self._rho * self._rho) - 3.0

    def __add__(self, that):
        """Add two Statistics objects together."""
        sigma = self.copy()
        sigma += that
        return sigma

    def __iadd__(self, that):
        """Add another Statistics object to this one."""
        cdef double sum_count = self._count + that._count
        if sum_count == 0:
            return self

        cdef double delta = that._eta - self._eta
        cdef double delta2 = delta ** 2
        cdef double delta3 = delta ** 3
        cdef double delta4 = delta ** 4

        cdef double sum_eta = (
            (self._count * self._eta + that._count * that._eta)
            / sum_count
        )

        cdef double sum_rho = (
            self._rho + that._rho
            + delta2 * self._count * that._count / sum_count
        )

        cdef double sum_tau = (
            self._tau + that._tau
            + delta3 * self._count * that._count
            * (self._count - that._count) / (sum_count ** 2)
            + 3.0 * delta
            * (self._count * that._rho - that._count * self._rho) / sum_count
        )

        cdef double sum_phi = (
            self._phi + that._phi
            + delta4 * self._count * that._count
            * (self._count ** 2 - self._count * that._count + that._count ** 2)
            / (sum_count ** 3)
            + 6.0 * delta2 * (
                self._count * self._count * that._rho
                + that._count * that._count * self._rho
            )
            / (sum_count ** 2)
            + 4.0 * delta
            * (self._count * that._tau - that._count * self._tau) / sum_count
        )

        if self._count == 0.0:
            self._min = that._min
            self._max = that._max
        elif that._count != 0.0:
            self._min = min(self._min, that._min)
            self._max = max(self._max, that._max)

        self._count = sum_count
        self._eta = sum_eta
        self._rho = sum_rho
        self._tau = sum_tau
        self._phi = sum_phi

        return self

    def __mul__(self, that):
        """Multiply by a scalar to change Statistics weighting."""
        if not isinstance(self, Statistics):
            self, that = that, self
        sigma = self.copy()
        sigma *= that
        return sigma

    def __imul__(self, that):
        """Multiply by a scalar to change Statistics weighting in-place."""
        that = float(that)
        self._count *= that
        self._rho *= that
        self._tau *= that
        self._phi *= that
        return self


cdef class Regression(object):
    """
    Compute simple linear regression in a single pass.

    Computes the slope, intercept, and correlation.
    Regression objects may also be added together and copied.

    Based entirely on the C++ code by John D Cook at
    http://www.johndcook.com/running_regression.html
    """
    cdef public Statistics _xstats
    cdef public Statistics _ystats
    cdef public double _count
    cdef public double _sxy

    def __init__(self, iterable=()):
        """Initialize Regression object.

        Iterates optional parameter `iterable` and pushes each pair into the
        regression summary.
        """
        self._xstats = Statistics()
        self._ystats = Statistics()
        self.clear()

        for xcoord, ycoord in iterable:
            self.push(xcoord, ycoord)

    cpdef clear(self):
        """Clear Regression object."""
        self._xstats.clear()
        self._ystats.clear()
        self._count = self._sxy = 0.0

    def __richcmp__(self, other, op):
        if op == 2:
            return self.get_state() == other.get_state()
        elif op == 3:
            return self.get_state() != other.get_state()
        else:
            return NotImplemented

    def get_state(self):
        """Get internal state."""
        return (
            self._count, self._sxy, self._xstats.get_state(),
            self._ystats.get_state()
        )

    def set_state(self, state):
        """Set internal state."""
        count, sxy, xstats, ystats = state
        self._count = count
        self._sxy = sxy
        self._xstats.set_state(xstats)
        self._ystats.set_state(ystats)

    @classmethod
    def fromstate(cls, state):
        """Return Regression object from state."""
        regr = cls()
        regr.set_state(state)
        return regr

    def __reduce__(self):
        return make_regression, (self.get_state(),)

    def copy(self, _=None):
        """Copy Regression object."""
        return self.fromstate(self.get_state())

    __copy__ = copy
    __deepcopy__ = copy

    def __len__(self):
        """Number of values that have been pushed."""
        return int(self._count)

    cpdef push(self, xcoord, ycoord):
        """Add a pair `(x, y)` to the Regression summary."""
        self._sxy += (
            (self._xstats.mean() - xcoord)
            * (self._ystats.mean() - ycoord)
            * self._count
            / (self._count + 1)
        )
        self._xstats.push(xcoord)
        self._ystats.push(ycoord)
        self._count += 1

    cpdef slope(self, ddof=1.0):
        """Slope of values (with `ddof` degrees of freedom)."""
        sxx = self._xstats.variance(ddof) * (self._count - ddof)
        return self._sxy / sxx

    cpdef intercept(self, ddof=1.0):
        """Intercept of values (with `ddof` degrees of freedom)."""
        return self._ystats.mean() - self.slope(ddof) * self._xstats.mean()

    cpdef correlation(self, ddof=1.0):
        """Correlation of values (with `ddof` degrees of freedom)."""
        term = self._xstats.stddev(ddof) * self._ystats.stddev(ddof)
        return self._sxy / ((self._count - ddof) * term)

    def __add__(self, that):
        """Add two Regression objects together."""
        sigma = self.copy()
        sigma += that
        return sigma

    def __iadd__(self, that):
        """Add another Regression object to this one."""
        sum_count = self._count + that._count
        if sum_count == 0:
            return self

        sum_xstats = self._xstats + that._xstats
        sum_ystats = self._ystats + that._ystats

        deltax = that._xstats.mean() - self._xstats.mean()
        deltay = that._ystats.mean() - self._ystats.mean()
        sum_sxy = (
            self._sxy + that._sxy
            + self._count * that._count * deltax * deltay / sum_count
        )

        self._count = sum_count
        self._xstats = sum_xstats
        self._ystats = sum_ystats
        self._sxy = sum_sxy

        return self
