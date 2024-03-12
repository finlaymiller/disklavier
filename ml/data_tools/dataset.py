from torch.utils.data import Dataset

from typing import List

from utils import console


class MIDILoopDataset(Dataset):
    def __init__(
        self,
        data,
        multiplier: int = 1,
        transforms=None,
    ):
        self.data = data * multiplier
        self.transforms = transforms

        console.log(
            f"[plum3]augment[/plum3]: created dataset with {len(self.data)} images"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, image = self.data[idx]
        if self.transforms:
            image = self.transforms(image)
        return name, image
