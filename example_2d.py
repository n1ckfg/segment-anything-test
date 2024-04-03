import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup("vit_b")

image_bgr = cv2.imread("data/biscarosse.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

sw.plot_image(image_rgb, result, "output/test_result.jpg")


