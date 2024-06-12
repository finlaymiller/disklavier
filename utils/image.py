import torch
from torch.nn.functional import pad

import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from utils import console

from typing import List
from numpy.typing import NDArray

p = "[navajo_white1]utils[/navajo_white1]  :"


def plot_images(
    images: List[NDArray],
    titles: List[str],
    main_title=None,
    outfile_name: str = f"{datetime.now().strftime('%y-%m-%d_%H%M%S')}",
    output_dir: str = "img",
    set_axis: str = "off",
) -> None:
    """Plot images vertically"""
    plt.style.use("dark_background")
    num_images = len(images)

    if main_title:
        plt.suptitle(main_title)
    for num_plot in range(num_images):
        plt.subplot(num_images, 1, num_plot + 1)
        plt.imshow(
            np.squeeze(images[num_plot]),
            aspect="auto",
            origin="lower",
            cmap="magma",
            interpolation="nearest",
        )
        plt.title(titles[num_plot])
        plt.axis(set_axis)

    plt.savefig(
        os.path.join(output_dir, outfile_name),
        dpi=300,
        bbox_inches="tight",
    )


def format_image(image: NDArray | torch.Tensor, remove_time=True) -> torch.Tensor:
    og_shape = image.shape

    # send image to cpu (required after training)
    if isinstance(image, torch.Tensor):
        image = image.cpu().data

    # remove time factor
    if remove_time:
        image = np.delete(image, 0, axis=1)

    # add 1 dimension & ensure correct data type
    image = torch.from_numpy(np.expand_dims(image, 0)).to(torch.float32)

    # normalize to [0, 1]
    if torch.any(image > 1.0):
        image = image / image.max()

    # zero-pad to (1, 140, 412)
    # padding is (left, right, top, bottom)
    image_width = image.shape[-1]
    padding_left = (412 - image_width) // 2
    padding_right = 412 - image_width - padding_left
    image = pad(image, (padding_left, padding_right, 6, 6), mode="constant", value=0.0)

    # console.log(f"{p} padded image {og_shape} -> {image.shape}")

    return image
