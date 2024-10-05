from tftb.processing import MargenauHillDistribution
import numpy as np


def test_tftb_margenauhill():
    """Test that tftb's MargenauHillDistribution works correctly."""
    n_samples = 256
    n_fbins = 128
    signal = np.random.randn(n_samples)

    tfr_real = MargenauHillDistribution(signal, n_fbins=n_fbins)
    tfr_real.run()
