"""Helpful tools for generating self-similarity matrices and novelty curves"""

import numpy as np
from numba import jit
from scipy import signal

import numpy.typing as npt
from typing import Tuple, Dict


def boundary_split(array, indices):
    """chatgpt-generated"""
    # Check if the indices are valid
    if any(index < 0 or index > len(array) for index in indices):
        raise ValueError("Invalid index in the list.")

    # Add the start and end indices to the list
    split_indices = [0] + sorted(indices) + [len(array)]

    # Create subarrays using slicing
    subarrays = [
        array[int(split_indices[i]) : int(split_indices[i + 1])]
        for i in range(len(split_indices) - 1)
    ]

    return subarrays


def boundary_split_t(array, times):
    subarrays = {}
    for t in times:
        subarrays[int(t)] = []
    np.append(times, 999999999)

    for note in array:
        for i, t in enumerate(times):
            if i == len(times) - 1:
                break
            if t < note.start and note.start < times[i + 1]:
                subarrays[int(t)].append(note)
                break

    return subarrays


def gen_ssm_and_novelty(
    features, L=1, filter_length=41, down_sampling=10, hop=1024, sr=44100
):
    features, _ = smooth_downsample_feature_sequence(
        features, sr / hop, filter_length, down_sampling
    )

    print(features.shape)

    ssm = np.dot(np.transpose(features), features)
    novelty = compute_novelty_ssm(ssm, L, exclude=True)

    return ssm, novelty


def get_peaks(data: npt.NDArray, Thalf=10, tau=1.35, distance=7):
    """from SSMNet"""

    nb_frame = len(data)

    # compute peak to mean ratio
    peak_to_mean_v = np.zeros((nb_frame))
    for nu in range(0, nb_frame):
        sss = max(0, nu - Thalf)
        eee = min(nu + Thalf + 1, nb_frame)
        local_mean = np.sum(data[sss:eee]) / (eee - sss)
        peak_to_mean_v[nu] = data[nu] / local_mean if local_mean != 0 else 0

    # find peaks
    peaks, _ = signal.find_peaks(peak_to_mean_v, distance=distance)

    # above threshold tau
    above_threshold = np.where(peak_to_mean_v[peaks] >= tau)[0]

    return peaks[above_threshold]


def get_boundaries(
    novelty: npt.NDArray[np.float64],
    time_sec_v: npt.NDArray[np.float64],
    peak_settings: Dict = {},
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Estimate the boundaries using the novelty curve.

    Args:
        novelty: the novelty curve
        time_sec_v: an array of seconds indicators
        peak_settings: custom settings for peak detection

    Returns:
        boundary_sec_v: boundary indices in seconds
        boundary_frame_v: boundary indices in frames
    """
    if len(peak_settings.values()) > 0:
        boundary_frame_v = get_peaks(
            novelty,
            peak_settings["Thalf"],
            peak_settings["tau"],
            peak_settings["distance"],
        )
    else:
        boundary_frame_v = get_peaks(
            novelty,
        )
    boundary_sec_v = time_sec_v[boundary_frame_v]

    # add start and end-time
    boundary_sec_v = np.concatenate(
        (0 * np.ones(1), boundary_sec_v, time_sec_v[-1] * np.ones(1))
    )

    # to be sure there is not twice zero
    boundary_sec_v = np.array(sorted([aaa for aaa in set(boundary_sec_v)]))

    return boundary_sec_v, boundary_frame_v


def compute_novelty_ssm(
    S, kernel=None, L=10, var=0.5, exclude=False
) -> npt.NDArray[np.float64]:
    """Compute novelty function from SSM [FMP, Section 4.4.1]

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        S (np.ndarray): SSM
        kernel (np.ndarray): Checkerboard kernel (if kernel==None, it will be computed) (Default value = None)
        L (int): Parameter specifying the kernel size M=2*L+1 (Default value = 10)
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 0.5)
        exclude (bool): Sets the first L and last L values of novelty function to zero (Default value = False)

    Returns:
        nov (np.ndarray): Novelty function
    """
    if kernel is None:
        kernel = compute_kernel_checkerboard_gaussian(L=L, var=var)
    N = S.shape[0]
    M = 2 * L + 1
    nov = np.zeros(N)
    # np.pad does not work with numba/jit
    S_padded = np.pad(S, L, mode="constant")

    for n in range(N):
        # Does not work with numba/jit
        nov[n] = np.sum(S_padded[n : n + M, n : n + M] * kernel)
    if exclude:
        right = np.min([L, N])
        left = np.max([0, N - L])
        nov[0:right] = 0
        nov[left:N] = 0

    return nov


@jit(nopython=True)
def compute_kernel_checkerboard_gaussian(
    L, var=1.0, normalize=True
) -> npt.NDArray[np.float64]:
    """Compute Guassian-like checkerboard kernel [FMP, Section 4.4.1].
    See also: https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/

    Notebook: C4/C4S4_NoveltySegmentation.ipynb

    Args:
        L (int): Parameter specifying the kernel size M=2*L+1
        var (float): Variance parameter determing the tapering (epsilon) (Default value = 1.0)
        normalize (bool): Normalize kernel (Default value = True)

    Returns:
        kernel (np.ndarray): Kernel matrix of size M x M
    """
    taper = np.sqrt(1 / 2) / (L * var)
    axis = np.arange(-L, L + 1)
    gaussian1D = np.exp(-(taper**2) * (axis**2))
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    kernel_box = np.outer(np.sign(axis), np.sign(axis))
    kernel = kernel_box * gaussian2D

    if normalize:
        kernel = kernel / np.sum(np.abs(kernel))

    return kernel


def smooth_downsample_feature_sequence(
    X, Fs, filt_len=41, down_sampling=10, w_type="boxcar"
):
    """Smoothes and downsamples a feature sequence. Smoothing is achieved by convolution with a filter kernel

    Notebook: C3/C3S1_FeatureSmoothing.ipynb

    Args:
        X (np.ndarray): Feature sequence
        Fs (scalar): Frame rate of ``X``
        filt_len (int): Length of smoothing filter (Default value = 41)
        down_sampling (int): Downsampling factor (Default value = 10)
        w_type (str): Window type of smoothing filter (Default value = 'boxcar')

    Returns:
        X_smooth (np.ndarray): Smoothed and downsampled feature sequence
        Fs_feature (scalar): Frame rate of ``X_smooth``
    """
    filt_kernel = np.expand_dims(signal.get_window(w_type, filt_len), axis=0)
    X_smooth = signal.convolve(X, filt_kernel, mode="same") / filt_len
    X_smooth = X_smooth[:, ::down_sampling]
    Fs_feature = Fs / down_sampling
    return X_smooth, Fs_feature
