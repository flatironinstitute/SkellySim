# Author: Max Kapur
# https://github.com/maxkapur/param_tools
# No license provided as of 7/19/21

import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import brentq


def arc_cumulator(t, coords):
    """
    Parameters
    ----------
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t.

    Returns
    -------
    t : array
        As above.
    cum_s : array
        Cumulative arc length on [t[0], t].

    Evaluates the cumulative arc length at each coordinate in sequence.
    """

    if np.all(t) is None:
        t = np.linspace(0, 1, coords.shape[-1])

    assert t.shape == coords.shape[1:], \
        "Need same number of parameters as coordinates"
    delta_s = np.linalg.norm(np.diff(coords), axis=0)
    cum_s = np.concatenate([[0], np.cumsum(delta_s)])

    return t, cum_s


def r_arc_from_data(n, t, coords, interp=True, kind='linear'):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t.
    interp : boolean
        Whether to generate random function values or not. Set to False if
        you have another way of evaluating the function.
    kind : str
        Interpolation method to be passed to scipy.interpolate.

    Returns
    -------
    (rand_coords : array)
        Random coordinates, if interp=True was passed.
    rand_t : array
        Parameters associated with the coordinates.
    rand_s : array
        Cumulative arc length at each coordinate.

    If the parameterizing function is known, use r_arc() instead, as the
    coordinates will be exactly computed from t instead of interpolating.
    """

    t, cum_s = arc_cumulator(t, coords)

    # Get random values on [0, length]
    # To test uniformity, you can use:
    # rand_s = np.linspace(0,cum_s[-1], n)
    rand_s = np.random.rand(n) * cum_s[-1]

    # Find corresponding t-values by interpolation
    rand_t = interp1d(cum_s, t)(rand_s)

    if interp:
        # Interpolate coordinates, e.g. if func unknown
        rand_coords = interp1d(t, coords, kind=kind)(rand_t)
        return rand_coords, rand_t, rand_s

    else:
        return rand_t, rand_s


def r_arc(n, func, t0, t1, precision=225):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1 : ints or floats
        Range over which func is evaluated.
    precision : int
        Number of t-values at which func is evaluated when computing
        arc length.

    Returns
    -------
    rand_coords : array
        Random coordinates.
    rand_t : array
        Parameters associated with the coordinates.
    rand_s : array
        Cumulative arc length at each coordinate.

    Generates random points distributed uniformly along a parametric curve.
    """

    t = np.linspace(t0, t1, precision)
    coords = func(t)

    rand_t, rand_s = r_arc_from_data(n, t, coords, interp=False)
    rand_coords = func(rand_t)

    return rand_coords, rand_t, rand_s


def arc_length(func, t0, t1, precision=225):
    """
    Parameters
    ----------
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1: ints or floats
        Range over which func is evaluated.
    precision : int
        Number of t-values at which func is evaluated.

    Returns
    -------
    length : float
        Estimate of surface area.

    Convenience function to evaluate total arc length over given range.
    """

    t = np.linspace(t0, t1, precision)
    coords = func(t)

    length = arc_cumulator(t, coords)[1][-1]

    return length


def sample_to_arc(sample, func, t0=0, precision=225, kind='linear', ub=1e11):
    """
    Parameters
    ----------
    sample : array
        Arc lengths at which to generate points.
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0 : float
        t-value to regard as having zero arc length.
    precision : int
        Number of t-values at which func is evaluated when computing
        arc length.
    kind : str
        Interpolation method to be passed to scipy.interp1d.
    ub : float
        Upper bound for t-values used when searching for extremes of
        distribution in brentq. An arbitrary large number.

    Returns
    -------
    sample_x : array
        Coordinates associated with sample arc lengths.
    sample_t : array
        Parameters associated with the coordinates.

    Maps a sample of arc lengths (e.g. np.random.normal(size=100)) to points
    along the curve given by func. Negative arc lengths are mapped to t-values
    below t0.

    This function is designed to take arbitrary samples as input. If you know
    the bounds of the sample data, arc_cumulator() followed by interp1d() will
    be more efficient (see examples).
    """
    # Separately compute the arc length on [0, t0] to offset sample
    # Passing func(t-t0) throws an error in brentq
    if t0 != 0:
        offset = arc_length(func, 0, t0, precision)
        sample = sample + offset * np.sign(t0)

    # Split sample into negative and positive components
    sign_idx = sample < 0
    sample_neg = np.abs(sample[sign_idx])
    sample_pos = sample[~sign_idx]

    # Compute extremes of sample
    s_min = np.abs(sample.min())
    s_max = sample.max()

    if sample_neg.size == 0:
        sample_t_neg = np.empty(0)
    else:
        # Find negative t-value at which Euclidean distance is s_min. By the triangle
        # inequality, the arc length is greater than s_min, so this t-range will be
        # wide enough to interpolate the whole sample.
        t_min = -brentq(lambda t: np.linalg.norm([func(-t), func(0)]) - s_min, 0, ub)

        # Perform the interpolation and generate t-values for sample arc lengths.
        arc_t_neg = np.linspace(0, t_min, precision)
        coords_neg = func(arc_t_neg)
        arc_cum_s_neg = arc_cumulator(arc_t_neg, coords_neg)[1]
        sample_t_neg = interp1d(arc_cum_s_neg, arc_t_neg, kind=kind)(sample_neg)

    if sample_pos.size == 0:
        sample_t_pos = np.empty(0)
    else:
        # As above, for positive component of sample
        t_max = brentq(lambda t: np.linalg.norm([func(t), func(0)]) - s_max, 0, ub)
        arc_t_pos = np.linspace(0, t_max, precision)
        coords_pos = func(arc_t_pos)
        arc_cum_s_pos = arc_cumulator(arc_t_pos, coords_pos)[1]
        sample_t_pos = interp1d(arc_cum_s_pos, arc_t_pos, kind=kind)(sample_pos)

    # Recombine negative and positive components and evaluate function.
    sample_t = np.empty_like(sample, dtype='float64')
    sample_t[sign_idx] = sample_t_neg
    sample_t[~sign_idx] = sample_t_pos
    sample_x = func(sample_t)

    return sample_x, sample_t


def surface_cumulator(t, u, coords):
    """
    Parameters
    ----------
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    u : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t, u pair.

    Returns
    -------
    t, u : arrays
        As above.
    cum_S_t : array
        Cumulative surface area on [t[0], t], all u.
    cum_S_u : array
        Cumulative surface area on all t, [u[0], u].

    Evaluates the cumulative surface area at each coordinate.
    """

    if np.all(t) is None:
        t, _ = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))
    if np.all(u) is None:
        _, u = np.meshgrid(np.linspace(0, 1, coords.shape[-2]),
                           np.linspace(0, 1, coords.shape[-1]))

    assert t.shape == u.shape == coords.shape[1:], \
        "Need same number of parameters as coordinates"
    delta_t_temp = np.diff(coords, axis=2)
    delta_u_temp = np.diff(coords, axis=1)

    # Pad with zeros so that small rand_S can still be interpd
    delta_t = np.zeros(coords.shape)
    delta_u = np.zeros(coords.shape)

    delta_t[:coords.shape[0], :coords.shape[1], 1:coords.shape[2]] = delta_t_temp
    delta_u[:coords.shape[0], 1:coords.shape[1], :coords.shape[2]] = delta_u_temp

    # Area of each parallelogram
    delta_S = np.linalg.norm(np.cross(delta_t, delta_u, 0, 0), axis=2)

    cum_S_t = np.cumsum(delta_S.sum(axis=0))
    cum_S_u = np.cumsum(delta_S.sum(axis=1))

    return t, u, cum_S_t, cum_S_u


def r_surface_from_data(n, t, u, coords, interp=True, kind='linear'):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    t : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    u : array or None
        Parameter values associated with data coordinates. Should be sorted
        already. Pass None to use np.linspace(0, 1, n).
    coords : array
        Values of the curve at each t, u pair.
    interp : boolean
        Whether to generate random function values or not. Set to false if you
        have another way of evaluating the function.
    kind : str
        Interpolation method to be passed to scipy.interpolate.

    Returns
    -------
    (rand_coords : array)
        Random coordinates, if interp=True was passed.
    rand_t : array
        t-values associated with the coordinates.
    rand_u : array
        u_values associated with the coordinates.
    rand_S_t : array
        Cumulative area at each t-value over full range of u.
    rand_S_u : array
        Cumulative area at each u-value over full range of t.

    If the parameterizing function is known, use r_surface() instead, as the
    coordinates will be exactly computed from t instead of interpolating.
    """

    t, u, cum_S_t, cum_S_u = surface_cumulator(t, u, coords)

    # Random values
    rand_S_t = np.random.rand(n) * cum_S_t[-1]
    rand_S_u = np.random.rand(n) * cum_S_u[-1]

    # Find corresponding t-values by interpolation
    rand_t = interp1d(cum_S_t, t[0, :])(rand_S_t)
    rand_u = interp1d(cum_S_u, u[:, 0])(rand_S_u)

    if interp:
        # Interpolate coordinates, e.g. if func unknown

        rand_coords = np.empty([coords.shape[0], n])

        # One axis at a time, or else scipy throws dim mismatch
        for i in range(coords.shape[0]):
            f = interp2d(t, u, coords[i], kind=kind)

            # One point at time, or else scipy does a meshgrid
            for j in range(n):
                rand_coords[i, j] = f(rand_t[j], rand_u[j])

        return rand_coords, rand_t, rand_u, rand_S_t, rand_S_u

    else:
        return rand_t, rand_u, rand_S_t, rand_S_u


def r_surface(n, func, t0, t1, u0, u1, t_precision=25, u_precision=25):
    """
    Parameters
    ----------
    n : int
        Number of points to generate.
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1, u0, u1 : ints or floats
        Range over which func is evaluated.
    t_precision, u_precision : ints
        Number of t-values at which func is evaluated when computing
        surface area.

    Returns
    -------
    rand_coords : array
        Random coordinates.
    rand_t : array
        t-values associated with the coordinates.
    rand_u : array
        u_values associated with the coordinates.
    rand_S_t : array
        Cumulative area at each t-value over full range of u.
    rand_S_u : array
        Cumulative area at each u-value over full range of t.

    Generates random points distributed uniformly over a parametric surface.
    """

    t, u = np.meshgrid(np.linspace(t0, t1, t_precision),
                       np.linspace(u0, u1, u_precision))
    coords = func(t, u)

    rand_t, rand_u, rand_S_t, rand_S_u = r_surface_from_data(n, t, u, coords, interp=False)
    rand_coords = func(rand_t, rand_u)

    return rand_coords, rand_t, rand_u, rand_S_t, rand_S_u


def surface_area(func, t0, t1, u0, u1, t_precision=25, u_precision=25):
    """
    Parameters
    ----------
    func : function
        Parametric function describing the curve on which points should
        be generated.
    t0, t1, u0, u1 : ints or floats
        Range over which func is evaluated.
    t_precision, u_precision : ints
        Number of t- and u-values at which func is evaluated.

    Returns
    -------
    area : float
        Estimate of surface area.

    Convenience function to evaluate total surface area over given range.
    """

    t, u = np.meshgrid(np.linspace(t0, t1, t_precision), np.linspace(u0, u1, u_precision))
    coords = func(t, u)

    area = surface_cumulator(t, u, coords)[3][-1]

    return area
