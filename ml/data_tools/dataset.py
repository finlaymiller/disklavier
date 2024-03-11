from torch.utils.data import Dataset

from typing import List


class MIDILoopDataset(Dataset):
    def __init__(
        self,
        data,
        multiplier: int = 1,
        transforms=None,
    ):
        self.data = data * multiplier
        self.transforms = transforms

        print(f"Created dataset with {len(self.data)} images")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, image = self.data[idx]
        if self.transforms:
            image = self.transforms(image)
        return name, image
