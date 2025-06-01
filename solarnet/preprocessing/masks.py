from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path as PolygonPath
from rasterio.features import rasterize
from shapely import Polygon
from tqdm import tqdm

from solarnet.preprocessing.utils import is_directory_not_empty

IMAGE_SIZES = {
    'Modesto': (5000, 5000),
    'Fresno': (5000, 5000),
    'Oxnard': (4000, 6000),
    'Stockton': (5000, 5000)
}


class MaskMaker:
    """This class looks for all files defined in the metadata, and
    produces masks for all of the .tif files saved there.
    These files will be saved in <org_folder>_mask/<org_filename>.npy

    Attributes:
        data_folder: pathlib.Path
            Path of the data folder, which should be set up as described in `data/README.md`
    """

    def __init__(self, data_folder: Path = Path('data')) -> None:
        self.data_folder = data_folder

    def _read_data(self) -> Tuple[defaultdict, dict]:
        metadata_folder = self.data_folder / 'metadata'

        print(f'Reading metadata from {metadata_folder}...')

        polygon_pixels = self._csv_to_dict_polygon_pixels(
            pd.read_csv(metadata_folder / 'polygonVertices_PixelCoordinates.csv')
        )
        # TODO: potentially filter on jaccard index
        polygon_images = self._csv_to_dict_image_names(
            pd.read_csv(metadata_folder / 'polygonDataExceptVertices.csv',
                        usecols=['polygon_id', 'city', 'image_name', 'jaccard_index']
                        )
        )

        # keep only those cities, which are available (i.e. aerial images are downloaded)
        cities = IMAGE_SIZES.keys()
        available_cities = [city for city in cities
                            if is_directory_not_empty(self.data_folder / city)]
        print(f'Available cities: {available_cities}')

        polygon_images = {city: polygon_images[city]
                          for city in available_cities if city in polygon_images}
        
        print(f'Found {len(polygon_images)} cities with aerial images.')

        return polygon_images, polygon_pixels

    def process(self) -> None:

        polygon_images, polygon_pixels = self._read_data()

        for city, aerial_image_files in polygon_images.items():
            print(f'Processing {city}')

            # first, we make sure the mask file exists; if not,
            # we make it
            masked_city = self.data_folder / f"{city}_masks"
            x_size, y_size = IMAGE_SIZES[city]

            if not masked_city.exists():
                masked_city.mkdir()

            for image, polygons in tqdm(aerial_image_files.items(), desc="Creating masks!"):
                mask = np.zeros((x_size, y_size), dtype=np.byte)

                for polygon in polygons:

                    polygon_coords = polygon_pixels[polygon]

                    if len(polygon_coords) < 3:
                        # skip in case it is a line or a point
                        continue

                    mask += self.make_mask_v2(polygon_coords, (x_size, y_size))

                np.save(masked_city / f"{image}.npy", mask)

    @staticmethod
    def _csv_to_dict_polygon_pixels(polygon_pixels: pd.DataFrame) -> dict:
        output_dict = {}

        for idx, row in polygon_pixels.iterrows():
            vertices = []
            for i in range(1, int(row.number_vertices) + 1):
                vertices.append((row[f"lat{i}"], row[f"lon{i}"]))
            output_dict[int(row.polygon_id)] = vertices
        return output_dict

    @staticmethod
    def _csv_to_dict_image_names(polygon_images: pd.DataFrame) -> defaultdict:
        output_dict: defaultdict = defaultdict(lambda: defaultdict(list))

        for idx, row in polygon_images.iterrows():
            output_dict[row.city][row.image_name].append(int(row.polygon_id))
        return output_dict

    @staticmethod
    def make_mask(coords: List, imsizes: Tuple[int, int]) -> np.array:
        """https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        """
        poly_path = PolygonPath(coords)

        x_size, y_size = imsizes
        x, y = np.mgrid[:x_size, :y_size]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        mask = poly_path.contains_points(coors)

        return mask.reshape(x_size, y_size).astype(float)

    @staticmethod
    def make_mask_v2(coords: List, imsizes: Tuple[int, int]) -> np.array:
        """Create masks from polygon vertices using `rasterio`
        """
        # the coords are in pixel coordinates (x: to the left, y: bottom)
        # however the rasterize function operates in matrix coordinates, i.e.
        # dimension 0: rows, dimension 1: columns - thus we need to invert
        coords = [(c[1], c[0]) for c in coords]
        
        poly = Polygon(coords)

        mask = rasterize([poly], out_shape=imsizes, fill=0, default_value=1,
                         dtype=np.byte, all_touched=False)
        return mask
