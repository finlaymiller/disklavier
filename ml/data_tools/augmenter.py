from torch.utils.data import Dataset

import os
import random
import numpy as np
from PIL import ImageFilter
from tqdm import tqdm

from typing import List

from data.dataset import MIDILoopDataset
from data.augmentation import vertical_shift, smear, noise
from utils.image import format_image


class DataAugmenter:
    dataset: Dataset
    dataset_name: str
    dataset_path: str

    def __init__(
        self,
        dir: str,
        params,
        default_dataset: str = "input_data/all_data.npz",
    ):
        self.dir = dir
        self.params = params
        self.default_set = default_dataset
        self.dataset_name = self.build_file_string()
        self.dataset_path = os.path.join(self.dir, self.dataset_name)

    def build_file_string(self) -> str:
        """terms are in the order that they are applied to the image"""

        v_str = "_v" if self.params.do_vshift else ""
        s_str = (
            f"_s-{self.params.smear_filter}-{self.params.smear_radius}"
            if self.params.do_smear
            else ""
        )
        n_str = (
            f"_n-{int(self.params.noise_factor * 100):03d}"
            if self.params.do_noise
            else ""
        )

        return f"ad_{self.params.factor}{v_str}{s_str}{n_str}"

    def get_clean(self, index: int | None = None):
        """return an image from the default input dataset, unless overfitting, then return from the actual dataset"""
        if os.path.exists(self.default_set) and not self.params.overfit:
            with np.load(self.default_set) as f:
                if index is None:
                    index = int(random.uniform(0, len(self._n2l(f))))
                    clean_image = self._n2l(f)[index]
        else:
            print(
                f"not getting clean image from default dataset, falling back on test set"
            )
            clean_image = self.dataset[0]

        return clean_image

    def load(self) -> bool:
        """Load dataset, first checking to see if one matching the current
        parameters already exists
        """
        dataset_found = False
        if os.path.exists(self.dataset_path + ".npz"):
            print(
                f"found a {os.path.getsize(self.dataset_path + '.npz')}B dataset matching the current parameters at\n\t{self.dataset_path}\nloading from there..."
            )
            with np.load(self.dataset_path + ".npz") as f:
                self.dataset = MIDILoopDataset(self._n2l(f))
            dataset_found = True
        else:
            print(
                f"no dataset found matching the current parameters\n\t{self.dataset_name}"
            )

        return dataset_found

    def save(self, data: List):
        """save out newly-augmented dataset to `.npz` file in same directory as input file."""

        print(f"saving dataset to \n\t{self.dataset_path}.npz\t")

        np.savez_compressed(
            self.dataset_path,
            **{name: arr for name, arr in data},
        )

        if os.path.exists(self.dataset_path + ".npz"):
            print(f"successfully wrote {os.path.getsize(self.dataset_path + '.npz')}B")
        else:
            raise FileNotFoundError

    def augment(self, force_rebuild: bool = False):
        """augments a set of passed-in images by a factor of factor^2"""

        if self.params.overfit:
            with np.load(self.default_set) as f:
                print(f"overfit test activated")
                data = self._n2l(f)
                random.shuffle(data)
                random_image = data[self.params.overfit_index]
                self.dataset = MIDILoopDataset(
                    [(random_image[0], format_image(random_image[1]))],
                    self.params.overfit_length,
                )
        elif not self.load():
            with np.load(self.default_set) as f:
                print(f"generating new augmentation from {self.default_set}")
                dataset_augmented = self._augment(self._n2l(f))
                self.save(dataset_augmented)
                self.dataset = MIDILoopDataset(dataset_augmented)

            # ensure that data was written correctly
            if not self.load():
                print(f"failed to load new dataset {self.dataset_path}")
                raise FileNotFoundError

        return self.dataset

    def _augment(
        self,
        clean_images: List,
    ):
        """internal method"""
        first_pass = []
        second_pass = []

        # shift images
        for name, image in tqdm(clean_images, unit="images", desc="shifting"):
            time_factor = image[:, 0]  # save manually included time factor
            image = np.delete(image, 0, axis=1)  # remove it from the image though
            image = image / np.max(image)  # normalize

            if self.params.do_vshift:
                # vertical shift images
                first_pass += vertical_shift(image, name, self.params.factor)
            else:
                # just reformat clean image array
                first_pass = [(name, image)]

            if self.params.do_smear:
                for i, si in enumerate(first_pass):
                    new_smear = smear(si[1], self._s2f(), self.params.smear_radius)
                    first_pass[i] = (si[0], new_smear)

        # add noise to images
        for ki in tqdm(first_pass, unit="images", desc="noising"):
            new_key, image = ki
            for _ in range(self.params.factor):
                noisy_image = noise(image, self.params.noise_factor)
                second_pass.append((new_key, noisy_image))

        random.shuffle(second_pass)

        print(
            f"used {len(clean_images)} images to generate {len(second_pass)} images of shape {second_pass[0][1].size()}"
        )

        return second_pass

    def _s2f(self):
        """filter picker for smearing"""
        match self.params.smear_filter:
            case "box":
                return ImageFilter.BoxBlur
            case "gauss":
                return ImageFilter.GaussianBlur
            case "median":
                return ImageFilter.MedianFilter
            case "min":
                return ImageFilter.MinFilter
            case "mode":
                return ImageFilter.ModeFilter
            case _:
                return ImageFilter.MaxFilter

    def _n2l(self, d: dict) -> List:
        """"""
        return list(d.items())
