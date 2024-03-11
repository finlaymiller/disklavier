from torch.utils.data import DataLoader, Dataset
import random

from data.dataset import MIDILoopDataset

from numpy.typing import NDArray


class CustomLoader(DataLoader):

    def __init__(self, data: MIDILoopDataset, params):
        self.data = data
        if params.overfit:
            self.data = MIDILoopDataset(
                [data[params.overfit_index]] * params.overfit_length
            )

        super().__init__(
            self.data,
            params.batch_size,
            params.shuffle,
            num_workers=params.num_workers,
        )
