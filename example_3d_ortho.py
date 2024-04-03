import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup("vit_b")

point_cloud, colors = sw.import_point_cloud("data/34FN2_18.las")

ortho_image = sw.cloud_to_ortho_image(point_cloud, colors, 1.5)

result = mask_generator.generate(ortho_image)

sw.plot_image(ortho_image, result, "output/34FN2_18_orthoimage.jpg")

