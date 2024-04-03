import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

argv = sys.argv
inputUrl = None
modelType = None
saveMemoryMode = False

try:
	argv = argv[argv.index("--") + 1:] # get all args after "--"
	modelType = argv[0]
	inputUrl = argv[1]
except:
	inputUrl = "data/ITC_BUILDING.las"
	modelType = "vit_b"
	print("Loading default file with " + modelType + ".")

mask_generator = sw.setup(modelType, saveMemoryMode)

'''las = laspy.read(inputUrl)

coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

r=(las.red/65535*255).astype(int)
g=(las.green/65535*255).astype(int)
b=(las.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()
'''

point_cloud, colors = sw.import_point_cloud(inputUrl, True)

resolution = 500

center_coordinates = [189, 60, 2]

spherical_image, mapping = sw.generate_spherical_image(center_coordinates, point_cloud, colors, resolution)

spherical_image_rgb = cv2.cvtColor(spherical_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("spherical_projection_preview.jpg", spherical_image_rgb)

result = mask_generator.generate(spherical_image)

fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72, np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
color_mask = sw.sam_masks(result)
plt.axis('off')

imgUrl = "output/spherical_projection_segmented.jpg"
plt.savefig(imgUrl)

image = cv2.imread(imgUrl)

modified_point_cloud = sw.color_point_cloud(image, point_cloud, mapping)

sw.export_point_cloud("output/pcd_results.las", modified_point_cloud)
