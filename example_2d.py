import numpy as np
import cv2

import sam_wrapper as sw

mask_generator = sw.setup()

image_bgr = cv2.imread("data/biscarosse.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

sw.plot_image(image_rgb, result, "output/example_2d_result.jpg")


