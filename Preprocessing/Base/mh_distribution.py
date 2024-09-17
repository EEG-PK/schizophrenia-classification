import numpy as np
import matplotlib.pyplot as plt
from tftb.processing import MargenauHillDistribution


def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def convert_to_image(mh_distribution, flip=True):
    img = scale_minmax(mh_distribution, 0, 255).astype(np.uint8)
    if flip:
        img = np.flip(img, axis=0)
    img = 255 - img  # invert. make black==more energy
    return img


# z tftb
def margenau_hill_distribution_image(signal, extend=True):
    tfr_real = MargenauHillDistribution(signal)
    tfr_real.run()
    # tfr_real.plot(show_tf=False, kind='cmap', sqmod=False, threshold=0)

    threshold = 0.05
    tfr_real.tfr = tfr_real.tfr[:(tfr_real.tfr.shape[0] // 2), :]
    _threshold = np.amax(tfr_real.tfr) * threshold
    tfr_real.tfr[tfr_real.tfr <= _threshold] = 0.0
    extent = (0, tfr_real.ts.max(), 0, 0.5)
    plt.imshow(tfr_real.tfr, aspect='auto', cmap='viridis', origin='lower', extent=extent)
    plt.show()
    image = convert_to_image(tfr_real.tfr, flip=False)
    plt.imshow(image, aspect='auto', cmap='gray', origin='lower', extent=extent)
    plt.show()

    if extend:
        return image, extent
    else:
        return image
