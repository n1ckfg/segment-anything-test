import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup("vit_b", False)

las = laspy.read("data/ITC_BUILDING.las")

coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

r=(las.red/65535*255).astype(int)
g=(las.green/65535*255).astype(int)
b=(las.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()
resolution = 500

center_coordinates = [189, 60, 2]

spherical_image, mapping = sw.generate_spherical_image(center_coordinates, point_cloud, colors, resolution)

result = mask_generator.generate(spherical_image)

fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72, np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
color_mask = sw.sam_masks(result)
plt.axis('off')

imgUrl = "output/ITC_BUILDING_spherical_projection_segmented.jpg"
plt.savefig(imgUrl)

image = cv2.imread(imgUrl)

modified_point_cloud = sw.color_point_cloud(image, point_cloud, mapping)

sw.export_point_cloud("output/pcd_results.las", modified_point_cloud)
