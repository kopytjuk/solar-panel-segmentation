from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2

# from solarnet.datasets.utils import normalize
from utils import normalize


class SegmenterInferenceDataset(Dataset):
    def __init__(self,
                 image_folder: Path | str,
                 normalize: bool = True,
                 device: str | None = "cpu",
                 imsize: tuple[int, int] = (224, 224)) -> None:

        self.device = device
        self.normalize = normalize
        self._imsize = imsize

        if isinstance(image_folder, str):
            image_folder = Path(image_folder)

        image_files = self.get_image_files(image_folder)
        image_files = sorted(image_files)
        self._image_files = image_files

    @staticmethod
    def get_image_files(folder: Path | str) -> List[str]:
        # Define the image file extensions you are interested in
        image_extensions = ['.png', '.jpg', '.jpeg', '.jp2', '.j2k']

        # Create a Path object for the folder
        if isinstance(folder, str):
            folder = Path(folder)

        # List to store the paths of image files
        image_files = []

        # Iterate over each file in the folder
        for file in folder.iterdir():
            # Check if the file has an image extension
            if file.suffix.lower() in image_extensions:
                image_files.append(file)

        return image_files

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, index: int) -> torch.Tensor:

        image_tensor = decode_image(self._image_files[index])

        image_tensor = v2.Resize(size=self._imsize)(image_tensor)

        # the following condition is needed, since MPS only supports float16
        if str(self.device) == "mps":
            image_tensor = v2.functional.to_dtype_image(image_tensor, dtype=torch.float16)

        image_tensor.to(self.device)

        if self.normalize:
            image_tensor = normalize(image_tensor)

        return image_tensor


if __name__ == "__main__":
    ds = SegmenterInferenceDataset("data/example_images")

    print(len(ds))

    first_image = ds[0]
    print(first_image.shape)
    print(first_image)
