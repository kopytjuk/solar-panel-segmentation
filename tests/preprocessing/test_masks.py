import pathlib
import random
import unittest
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from solarnet.preprocessing.masks import IMAGE_SIZES, MaskMaker


class TestMasks:

    def test_make_mask(self):
        """Ensure the masks are correctly generated
        """
        x_min = y_min = 2
        x_max = y_max = 4
        coords = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)]
        imsizes = (10, 10)

        mask = MaskMaker.make_mask(coords, imsizes)

        assert mask.shape == imsizes, f'Got mask shape {mask.shape}, expected {imsizes}'

        # make sure the shape is right.
        inclusion_zone = mask[x_min + 1: x_max, y_min + 1: y_max + 1]
        assert (inclusion_zone == 1).all(), 'Got 0-valued pixels within the masked area'

        exclusion_zones = [
            mask[:x_min + 1, :], mask[:, :y_min + 1], mask[x_max:], mask[:, y_max + 1:]
        ]
        for zone in exclusion_zones:
            assert (zone == 0).all(), 'Got 1-valued pixels outside the masked area'

    def test_make_mask_v2(self):
        """Ensure the masks are correctly generated
        """

        # the polygon is a square
        x_min = y_min = 2
        x_max = y_max = 4

        coords = [(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min), (x_min, y_min)]
        imsizes = (10, 10)

        mask = MaskMaker.make_mask_v2(coords, imsizes)

        # make sure the shape is right
        assert mask.shape == imsizes, f'Got mask shape {mask.shape}, expected {imsizes}'

        inclusion_zone = mask[x_min: x_max, y_min:y_max]
        assert (inclusion_zone == 1).all(
        ), 'Got 0-valued pixels within the masked area'

        exclusion_zones = [
            mask[:x_min, :], mask[:, :y_min], mask[x_max:], mask[:, y_max:]
        ]
        for zone in exclusion_zones:
            assert (zone == 0).all(), 'Got 1-valued pixels outside the masked area'

    def test_csv_to_polygon_pixels(self):
        """Test variable numbers of vertices are correctly handled
        """
        polygon_shapes = {0: 4, 1: 6, 2: 3}
        test_data = self._make_polygon_vertices_pixel_coordinates(polygon_shapes)

        # run the method
        polygon_dict = MaskMaker._csv_to_dict_polygon_pixels(test_data)

        for polygon_id, vertices in polygon_dict.items():
            assert len(vertices) == polygon_shapes[polygon_id], \
                f'Got {len(vertices)} for polygon {polygon_id}, ' \
                f'expected {polygon_shapes[polygon_id]}'

    @unittest.skip("Skipping")
    def test_process(self, tmp_path: pathlib.Path):
        """Test the process runs end to end
        """
        # setup
        polygon_shapes = {0: 4, 1: 6, 2: 3}  # polygon_idx -> num_vertices
        polygon_city_filenames = {0: ['Fresno', '1a'], 1: ['Oxnard', '2b'], 2: ['Modesto', '3c']}
        vertices_df = self._make_polygon_vertices_pixel_coordinates(polygon_shapes)
        except_vertices_df = self._make_polygon_data_except_vertices(polygon_city_filenames)

        metadata_path = tmp_path / "metadata"
        metadata_path.mkdir()
        vertices_df.to_csv(metadata_path / 'polygonVertices_PixelCoordinates.csv')
        except_vertices_df.to_csv(metadata_path / 'polygonDataExceptVertices.csv')

        # NOTE: this test is skipped, because MaskMaker expects the image data
        # to be present in the test folder,
        # which is not the case in this test environment.
        mask_maker = MaskMaker(tmp_path)
        mask_maker.process()

        for idx, (city, filename) in polygon_city_filenames.items():
            file_location = tmp_path / f'{city}_masks' / f'{filename}.npy'
            assert file_location.exists(), f'{file_location} not saved'

            mask = np.load(file_location)
            assert mask.shape == IMAGE_SIZES[city], \
                f'Got {mask.shape} for {city}, expected {IMAGE_SIZES[city]}'

    @classmethod
    def _make_polygon_vertices_pixel_coordinates(cls, polygon_shapes: Dict[int, int]):
        # make the fake data
        max_vertices = max(polygon_shapes.values())

        test_data = defaultdict(list)

        for polygon_idx, num_vertices in polygon_shapes.items():
            test_data['polygon_id'].append(polygon_idx)
            test_data['number_vertices'].append(num_vertices)

            polygon = cls._generate_random_polygon(num_vertices, area=(1000, 1000), seed=42)
            polygon_coords = list(polygon.exterior.coords)

            for vertex_nr in range(1, max_vertices+1):

                if vertex_nr > num_vertices:
                    # is filled, when the vertex number does not exist
                    vertex_x_value = None
                    vertex_y_value = None
                else:
                    coord = polygon_coords[vertex_nr-1]
                    vertex_x_value, vertex_y_value = coord

                test_data[f'lon{vertex_nr}'].append(vertex_x_value)
                test_data[f'lat{vertex_nr}'].append(vertex_y_value)

        return pd.DataFrame(data=test_data)

    @staticmethod
    def _make_polygon_data_except_vertices(polygon_cities_filenames):
        """Currently, Jaccard indices are not used, so that value
        can be kept static
        """
        test_data = defaultdict(list)
        for polygon, (city, filename) in polygon_cities_filenames.items():
            test_data['polygon_id'].append(polygon)
            test_data['city'].append(city)
            test_data['image_name'].append(filename)
            test_data['jaccard_index'].append(0.9)

        return pd.DataFrame(data=test_data)

    @staticmethod
    def _generate_random_polygon(num_edges: int, area: Tuple[int, int], max_attempts: int = 10,
                                 seed: int = 42) -> Polygon:
        """Given a number of edges generate a valid polygon in a restricted area.
        """

        random.seed(seed)

        # Ensure the number of edges is at least 3
        if num_edges < 3:
            raise ValueError("A polygon must have at least 3 edges.")

        for _ in range(max_attempts):
            # Generate random vertices
            vertices = []
            for _ in range(num_edges):
                x = random.uniform(0, area[0])
                y = random.uniform(0, area[1])
                vertices.append((x, y))

            # Create a polygon from the vertices
            polygon = Polygon(vertices)

            # Ensure the polygon is oriented counter-clockwise
            polygon = orient(polygon, sign=1.0)

            # Check if the polygon is valid
            if polygon.is_valid:
                return polygon

        raise ValueError("Unable to generate a valid polygon after multiple attempts.")
