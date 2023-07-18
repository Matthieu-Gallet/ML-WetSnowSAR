import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d


def pad_image(image, window_size):
    """Perform padding on an image to enable extraction of windows of size W x W

    Parameters
    ----------
    image : numpy array
        Image of dimension Nx, Ny, C

    window_size : int
        Size of the windows (square)

    Returns
    -------
    numpy array
        Padded image
    """

    height, width, _ = image.shape

    pad_height = (window_size - (height % window_size)) % window_size
    pad_width = (window_size - (width % window_size)) % window_size

    padded_image = np.pad(
        image, ((0, pad_height), (0, pad_width), (0, 0)), mode="constant"
    )

    return padded_image


def create_position_matrix(image_size):
    """Create a position matrix for an image of size Nx, Ny

    Parameters
    ----------
    image_size : tuple
        Size of the image (Nx, Ny)

    Returns
    -------
    numpy array
        Position matrix of size Nx, Ny, 2
    """
    Nx, Ny = image_size
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y)
    positions = np.stack((X, Y), axis=-1)
    positions = np.moveaxis(positions, 1, 0)
    return positions


def reconstruct_image_from_data(data, positions, image_size):
    """Reconstruct the original image from data and positions

    Parameters
    ----------
    data : numpy array
        Array of data

    positions : numpy array
        Array of positions corresponding to the data

    image_size : tuple
        Size of the original image (Nx, Ny, C)

    Returns
    -------
    numpy array
        Reconstructed image
    """
    reconstructed_image = np.zeros(image_size[:2])
    reconstructed_image[positions[:, 0], positions[:, 1]] = data

    return reconstructed_image


class Patch_extractor:
    """Class for managing the extraction of overlapping patches from an image
    and the reconstruction of the image from the patches.
    It is based on the scikit-learn class
    sklearn.feature_extraction.image.extract_patches_2d
    For the reconstruction, in case of odd window size, the center pixel
    of the patch is selected as the center pixel of the patch. In case of even
    window size, the upper left pixel in the center is selected as the center
    pixel of the patch.

    Attributes
    ----------
    window_size : int
        Size of the window (square)

    channel : list
        Channel to extract

    padding : bool
        Perform padding on the image before extracting patches

    positions_ : numpy array
        Array of positions corresponding to the extracted patches

    patches : numpy array
        Array of extracted patches

    image_size : tuple
        Size of the original image (Nx, Ny, C)
    """

    def __init__(self, window_size, channel, padding=True):
        self.window_size = window_size
        self.padding = padding
        self.channel = channel
        self.positions_ = None
        self.patches = None
        self.image_size = None

    def _prepare(self, image):
        if len(image.shape) != 3:
            raise ValueError("Image must be a 3D array")
        if self.padding:
            X = pad_image(image, self.window_size)
        else:
            X = image

        X = X[:, :, self.channel]
        self.image_size = X.shape
        return X

    def extract(self, image):
        X = self._prepare(image)
        positions = create_position_matrix(X.shape[:2])
        patches = extract_patches_2d(
            X, (self.window_size, self.window_size), random_state=0
        )
        positions_patch = extract_patches_2d(
            positions, (self.window_size, self.window_size), random_state=0
        )
        # strategy to select the center pixel of the patch or the upper left
        # pixel in the center in case of even window size
        self.positions_ = positions_patch[
            :,
            int((self.window_size / 2) + 0.5) - 1,
            int((self.window_size / 2) + 0.5) - 1,
            :,
        ]
        self.patches = np.array(patches, dtype=np.float32)
        return self.patches

    def reconstruct(self, data):
        return reconstruct_image_from_data(data, self.positions_, self.image_size)


def scale_data(img, axes=None, no_data=-999):
    """scale data between 0 and 1

    Parameters
    ----------
    img : np.array
        image to scale of shape (..., C), where C is the number of channels
        must the last dimension
    axes : list, optional
        axes to use for scaling, by default None
        None means all axes except the last one (channels) is used
    no_data : int, optional
        value to ignore, by default -999
        and replace by np.nan, it should be the minimum value of the image

    Returns
    -------
    np.array
        scaled image
    """

    if np.nanmin(img) == no_data:
        img[img == no_data] = np.nan
    if axes is None:
        axes = tuple(range(img.ndim - 1))
    if img.ndim == 2:
        axes = (0, 1)
    min_ = np.nanmin(img, axis=axes, keepdims=True)
    max_ = np.nanmax(img, axis=axes, keepdims=True)
    scale = max_ - min_
    out = (img - min_) / scale
    return out


# TODO : test
# for i in range(25):
#     img = np.random.rand(100, 100, 3)
#     w = 5
#     d = Patch_extractor(w, [0, 1, 2], padding=False)
#     patches = d.extract(img)
#     transformed = patches.mean(axis=-1)[:,2,2]
#     s = d.reconstruct(transformed)
#     np.testing.assert_almost_equal(img[w:-w, w:-w].mean(axis=-1), s[w:-w, w:-w])
