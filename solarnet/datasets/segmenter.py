import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

from .transforms import colour_jitter, horizontal_flip, no_change, vertical_flip
from .utils import normalize


class SegmenterDataset:
    def __init__(self,
                 processed_folder: Path = Path('data/processed'),
                 normalize: bool = True, transform_images: bool = True,
                 device: torch.device = torch.device('cuda:0' if
                                                     torch.cuda.is_available() else 'cpu'),
                 mask: Optional[List[bool]] = None) -> None:

        self.device = device
        self.normalize = normalize
        self.transform_images = transform_images

        # We will only segment the images which we know have solar panels in them; the
        # other images should be filtered out by the classifier
        solar_folder = processed_folder / 'solar'

        self.org_solar_files = list((solar_folder / 'org').glob("*.npy"))
        if not any((solar_folder / 'mask').iterdir()):
            self.mask_solar_files = []
        else:
            self.mask_solar_files = [solar_folder / 'mask' / f.name for f in self.org_solar_files]

        if mask is not None:
            self.add_mask(mask)

    def add_mask(self, mask: List[bool]) -> None:
        """Add a mask to the data
        """
        assert len(mask) == len(self.org_solar_files), \
            f"Mask is the wrong size! Expected {len(self.org_solar_files)}, got {len(mask)}"
        self.org_solar_files = [x for include, x in zip(mask, self.org_solar_files) if include]
        self.mask_solar_files = [x for include, x in zip(mask, self.mask_solar_files) if include]

    def __len__(self) -> int:
        return len(self.org_solar_files)

    def _transform_images(self, image: np.ndarray,
                          mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        transforms = [
            no_change,
            horizontal_flip,
            vertical_flip,
            colour_jitter,
        ]
        chosen_function = random.choice(transforms)
        return chosen_function(image, mask)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = np.load(self.org_solar_files[index])

        if len(self.mask_solar_files) != 0:
            y = np.load(self.mask_solar_files[index])
        else:
            y = np.zeros((0))
        
        # default tensor type
        dtype_tensor = torch.float32

        # the following condition is needed, since MPS only supports float16
        if str(self.device) == "mps":
            x = x.astype(np.float16)
            y = y.astype(np.float16)
            dtype_tensor = torch.float16

        if self.normalize:
            x = normalize(x)

        if len(self.mask_solar_files) != 0:
            if self.transform_images:
                x, y = self._transform_images(x, y)
            return torch.as_tensor(x.copy(), device=self.device, dtype=dtype_tensor).float(), \
                torch.as_tensor(y.copy(), device=self.device, dtype=dtype_tensor).float()
        else:  # if no masks area available, return only the original solar files:
            if self.transform_images:
                x = self._transform_images(x)
            return torch.as_tensor(x.copy(), device=self.device, dtype=dtype_tensor).float(), \
                torch.tensor([], device=self.device, dtype=dtype_tensor)
