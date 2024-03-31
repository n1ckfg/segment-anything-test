import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup()

# Reading the point cloud with laspy
pcd = laspy.read("data/34FN2_18.las")

# Transforming the point cloud to Numpy
pcd_np = np.vstack((pcd.x, pcd.y, pcd.z, (pcd.red/65535*255).astype(int),
(pcd.green/65535*255).astype(int),
(pcd.blue/65535*255).astype(int))).transpose()

# Ortho-Projection
ortho_image = sw.cloud_to_image(pcd_np, 1.5)

result = mask_generator.generate(ortho_image)

# Plotting and exporting
fig = plt.figure(figsize=(np.shape(ortho_image)[1]/72,
np.shape(ortho_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(ortho_image)
color_mask = sw.sam_masks(result)
plt.axis('off')
plt.savefig("output/34FN2_18_orthoimage.jpg")