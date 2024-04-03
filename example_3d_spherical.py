import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup("vit_b")

# Loading the las file from the disk
point_cloud, colors = sw.import_point_cloud("data/ITC_BUILDING.las")

resolution = 500

# Defining the position in the point cloud to generate a panorama
center_coordinates = [189, 60, 2]

spherical_image, mapping = sw.generate_spherical_image(point_cloud, colors, resolution, center_coordinates)

result = mask_generator.generate(spherical_image)

sw.plot_image(spherical_image, result, "output/ITC_BUILDING_spherical_projection_segmented.jpg")

