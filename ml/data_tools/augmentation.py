import torch

import random
import numpy as np
from pathlib import Path
from PIL import Image

from utils.image import format_image

from typing import List
from numpy.typing import NDArray


def vertical_shift(array, name: str, num_iterations: int = 1) -> List[NDArray]:
    """vertically shift an image
    NOTE: in some sense, up and down is flipped.
    """
    shifted_images = []

    def find_non_zero_bounds(arr):
        """Find the first and last row index with a non-zero element"""
        rows_with_non_zero = np.where(arr.any(axis=1))[0]
        return rows_with_non_zero[0], rows_with_non_zero[-1]

    def shift_array(arr, up=0, down=0):
        """Shift array vertically within bounds"""
        if up > 0:
            arr = np.roll(arr, -up, axis=0)
            arr[-up:] = 0
        elif down > 0:
            arr = np.roll(arr, down, axis=0)
            arr[:down] = 0
        return arr

    highest, lowest = find_non_zero_bounds(array)
    maximum_up = highest
    maximum_down = array.shape[0] - lowest - 1

    for _ in range(num_iterations):
        # Shift up and then down, decreasing the shift amount in each iteration
        for i in range(maximum_up, 0, -1):
            new_key = f"{Path(name).stem}_u{i:02d}"
            shifted_images.append((new_key, np.copy(shift_array(array, up=i))))
        for i in range(maximum_down, 0, -1):
            new_key = f"{Path(name).stem}_d{i:02d}"
            shifted_images.append((new_key, np.copy(shift_array(array, down=i))))

    random.shuffle(shifted_images)

    return shifted_images[:num_iterations]


def smear(array: NDArray, blur, radius: int = 5) -> NDArray:
    """uses PIL to apply a filter to an image"""
    image = Image.fromarray(array * 255).convert("L")
    image = image.filter(blur(radius))

    return np.asarray(image)


def noise(
    image: NDArray, noise_factor: float = 0.05, reformat: bool = True
) -> torch.Tensor:
    """adds noise to an image and formats it"""
    image = image / np.max(image)
    noisy_image = torch.from_numpy(image) + noise_factor * torch.randn(image.shape)
    if reformat:
        noisy_image = format_image(noisy_image, remove_time=False)

    return noisy_image
