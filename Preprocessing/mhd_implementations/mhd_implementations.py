import numpy as np
from scipy.signal import get_window
from scipy.fft import fft
from typing import Optional, Tuple

from tftb.processing import MargenauHillDistribution
from tftb.processing.cohen import PseudoMargenauHillDistribution


def margenau_hill_distribution_spectrogram_tfrmhs(
    x: np.ndarray,
    t: Optional[np.ndarray] = None,
    N: Optional[int] = None,
    g: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    ptrace: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Margenau-Hill distribution for a given signal.

    :param x: Input signal, expected to be a 1D array or 2D column vector.
    :type x: np.ndarray
    :param t: Time indices at which the distribution is computed, defaults to None.
    :type t: Optional[np.ndarray], optional
    :param N: Number of frequency bins, defaults to the length of the input signal.
    :type N: Optional[int], optional
    :param g: Smoothing window for the time domain, defaults to a Hanning window of length Nx//10.
    :type g: Optional[np.ndarray], optional
    :param h: Smoothing window for the frequency domain, defaults to a Hanning window of length Nx//4.
    :type h: Optional[np.ndarray], optional
    :param ptrace: If True, prints the progress of the computation, defaults to False.
    :type ptrace: bool, optional
    :return: A tuple containing the Margenau-Hill distribution matrix and the time indices.
    :rtype: Tuple[np.ndarray, np.ndarray]

    This function calculates the Margenau-Hill distribution spectrogram for the given signal. It handles
    optional inputs for time indices, frequency bins, and smoothing windows. The distribution is computed
    using a combination of windowed Fourier transforms and time-frequency smoothing.

    **Example usage**::

        tfr, t = margenau_hill_distribution_spectrogram_tfrmhs(signal, ptrace=True)

    """

    # Handling x input
    x = np.atleast_2d(x).T  # Ensure x is a column vector
    Nx, xcol = x.shape

    # Handle h input
    if h is None:
        hlength = Nx // 4
        if hlength % 2 == 0:
            hlength += 1
        h = get_window('hann', hlength)
    else:
        h = np.asarray(h)
        if h.ndim != 1 or not np.isreal(h).all():
            raise ValueError("Wrong type for argument #5: A real vector expected.")
        hlength = len(h)
        if hlength % 2 == 0:
            raise ValueError("Wrong size for argument #5: An odd number of elements expected.")
    h = h / h[(hlength - 1) // 2]

    # Handle g input
    if g is None:
        glength = Nx // 10
        if glength % 2 == 0:
            glength += 1
        g = get_window('hann', glength)
    else:
        g = np.asarray(g)
        if g.ndim != 1 or not np.isreal(g).all():
            raise ValueError("Wrong type for argument #4: A real vector expected.")
        glength = len(g)
        if glength % 2 == 0:
            raise ValueError("Wrong size for argument #4: An odd number of elements expected.")

    # Handle N input
    if N is None:
        N = Nx
    else:
        if not isinstance(N, int) or N <= 0:
            raise ValueError("Wrong type for input argument #3: A positive integer expected.")

    # Handle t input
    if t is None:
        t = np.arange(1, Nx + 1)
    else:
        t = np.asarray(t)
        if not np.isreal(t).all() or t.ndim != 1:
            raise ValueError("Wrong type for input argument #2: A real vector expected.")
        if np.any(t > Nx) or np.any(t < 1):
            raise ValueError(f"Wrong value for input argument #2: The elements must be in the interval [1, {Nx}].")
    Nt = len(t)

    Lg = (glength - 1) // 2
    Lh = (hlength - 1) // 2

    Lgh = min(Lg, Lh)
    points = np.arange(-Lgh, Lgh + 1)
    Kgh = np.sum(h[Lh + points] * g[Lg + points])
    h = h / Kgh

    tfr = np.zeros((N, Nt), dtype=complex)
    tfr2 = np.zeros((N, Nt), dtype=complex)

    if ptrace:
        print('Pseudo Margenau-Hill distribution')

    Nfs2 = N // 2

    for icol in range(Nt):
        ti = t[icol] - 1  # Adjust for Python's 0-based indexing

        if ptrace:
            print(f"Processing column {icol + 1}/{Nt}")

        tau = np.arange(-min(Nfs2, Lg, ti), min(Nfs2 - 1, Lg, Nx - ti - 1) + 1)
        indices = np.mod(N + tau, N)
        tfr[indices, icol] = x[ti + tau, 0] * g[Lg + tau]

        tau = np.arange(-min(Nfs2, Lh, ti), min(Nfs2 - 1, Lh, Nx - ti - 1) + 1)
        indices = np.mod(N + tau, N)
        tfr2[indices, icol] = x[ti + tau, xcol - 1] * np.conj(h[Lh + tau])

    tfr = np.real(fft(tfr, axis=0) * np.conj(fft(tfr2, axis=0)))

    return tfr, t


def margenau_hill_distribution_spectrogram_tfrmhs_ifft(
    x: np.ndarray,
    t: Optional[np.ndarray] = None,
    N: Optional[int] = None,
    g: Optional[np.ndarray] = None,
    h: Optional[np.ndarray] = None,
    ptrace: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Margenau-Hill distribution for a given signal.

    :param x: Input signal, expected to be a 1D array or 2D column vector.
    :type x: np.ndarray
    :param t: Time indices at which the distribution is computed, defaults to None.
    :type t: Optional[np.ndarray], optional
    :param N: Number of frequency bins, defaults to the length of the input signal.
    :type N: Optional[int], optional
    :param g: Smoothing window for the time domain, defaults to a Hanning window of length Nx//10.
    :type g: Optional[np.ndarray], optional
    :param h: Smoothing window for the frequency domain, defaults to a Hanning window of length Nx//4.
    :type h: Optional[np.ndarray], optional
    :param ptrace: If True, prints the progress of the computation, defaults to False.
    :type ptrace: bool, optional
    :return: A tuple containing the Margenau-Hill distribution matrix and the time indices.
    :rtype: Tuple[np.ndarray, np.ndarray]

    This function calculates the Margenau-Hill distribution spectrogram for the given signal. It handles
    optional inputs for time indices, frequency bins, and smoothing windows. The distribution is computed
    using a combination of windowed Fourier transforms and time-frequency smoothing.

    **Example usage**::

        tfr, t = margenau_hill_distribution_spectrogram_tfrmhs_ifft(signal, ptrace=True)

    """
    fname = "tfrmhs"

    # Input validation and defaults
    if x.ndim != 1 and x.ndim != 2:
        raise ValueError(f"{fname}: Wrong type for input argument #1: A matrix of double expected.")

    if x.ndim == 1:
        x = x[:, np.newaxis]
    Nx, xcol = x.shape

    if ptrace not in [True, False]:
        raise ValueError(f"{fname}: Wrong type for argument #6: A boolean or real scalar expected.")

    if h is None:
        hlength = Nx // 4
        if hlength % 2 == 0:
            hlength += 1
        h = np.hamming(hlength)
    else:
        if h.ndim != 1 or not np.isreal(h).all():
            raise ValueError(f"{fname}: Wrong type for argument #5: A real vector expected.")
        hlength = h.size
        if hlength % 2 == 0:
            raise ValueError(f"{fname}: Wrong size for argument #5: An odd number of elements expected.")

    h = h.flatten()
    Lh = (hlength - 1) // 2
    h = h / h[Lh]

    if g is None:
        glength = Nx // 10
        if glength % 2 == 0:
            glength += 1
        g = np.hamming(glength)
    else:
        if g.ndim != 1 or not np.isreal(g).all():
            raise ValueError(f"{fname}: Wrong type for argument #4: A real vector expected.")
        glength = g.size
        if glength % 2 == 0:
            raise ValueError(f"{fname}: Wrong size for argument #4: An odd number of elements expected.")

    g = g.flatten()
    Lg = (glength - 1) // 2

    if N is None:
        N = Nx
    else:
        if not (isinstance(N, int) and N > 0):
            raise ValueError(f"{fname}: Wrong value for input argument #3: A positive integer expected.")

    if t is None:
        t = np.arange(1, Nx + 1)
    else:
        if t.ndim != 1 or not np.isreal(t).all() or np.max(t) > Nx or np.min(t) < 1:
            raise ValueError(f"{fname}: Wrong value for input argument #2: The elements must be in the interval [1, {Nx}].")

    Nt = t.size
    Lgh = min(Lg, Lh)
    points = np.arange(-Lgh, Lgh + 1)
    Kgh = np.sum(h[Lh + points] * g[Lg + points])
    h = h / Kgh

    tfr = np.zeros((N, Nt), dtype=complex)
    tfr2 = np.zeros((N, Nt), dtype=complex)

    if ptrace:
        print('Pseudo Margenau-Hill distribution')

    Nfs2 = N // 2

    for icol in range(Nt):
        ti = t[icol] - 1  # Adjust for zero-based indexing
        if ptrace:
            print(f'Progress: {icol + 1}/{Nt}', end='\r')

        tau = np.arange(-min(Nfs2, Lg, ti), min(Nfs2, Lg, Nx - ti))
        indices = np.mod(N + tau, N)
        tfr[indices, icol] = x[ti + tau, 0] * g[Lg + tau]

        tau = np.arange(-min(Nfs2, Lh, ti), min(Nfs2, Lh, Nx - ti))
        indices = np.mod(N + tau, N)
        tfr2[indices, icol] = x[ti + tau, xcol - 1] * np.conj(h[Lh + tau])

    tfr = np.real(np.fft.ifft(tfr, axis=0) * np.conj(np.fft.ifft(tfr2, axis=0)))
    return tfr, t


def margenau_hill_distribution(signal: np.ndarray, n_fbins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Margenau-Hill distribution of a signal.

    :param signal: The input signal for which the distribution is computed.
    :type signal: np.ndarray
    :param n_fbins: The number of frequency bins to use for the distribution, defaults to None.
    :type n_fbins: int, optional
    :return: A tuple containing the time-frequency representation (TFR) and the time instants.
    :rtype: Tuple[np.ndarray, np.ndarray]

    This function computes the Margenau-Hill distribution using the `tftb` library. It returns the TFR and
    the corresponding time instants.

    **Example usage**::

        tfr, ts = margenau_hill_distribution(signal)

    """
    tfr_real = MargenauHillDistribution(signal, n_fbins=n_fbins)
    tfr_real.run()

    return tfr_real.tfr, tfr_real.ts


def pseudo_margenau_hill_distribution(signal: np.ndarray, n_fbins: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Pseudo-Margenau-Hill distribution of a signal.

    :param signal: The input signal for which the distribution is computed.
    :type signal: np.ndarray
    :param n_fbins: The number of frequency bins to use for the distribution, defaults to None.
    :type n_fbins: int, optional
    :return: A tuple containing the real part of the time-frequency representation (TFR) and the time instants.
    :rtype: Tuple[np.ndarray, np.ndarray]

    This function computes the Pseudo-Margenau-Hill distribution using the `tftb` library. It returns the real part of
    the TFR and the corresponding time instants.

    **Example usage**::

        tfr, ts = pseudo_margenau_hill_distribution(signal)

    """
    pmhd = PseudoMargenauHillDistribution(signal, n_fbins=n_fbins)
    pmhd.run()

    return np.real(pmhd.tfr), pmhd.ts
