import numpy as np
import cv2

import sam_wrapper as sw

mask_generator = sw.setup()

point_cloud, colors = sw.import_point_cloud("data/34FN2_18.las")

ortho_image = sw.cloud_to_ortho_image(point_cloud, colors, 1.5)

result = mask_generator.generate(ortho_image)

sw.plot_image(ortho_image, result, "output/example_3d_ortho_result.jpg")

