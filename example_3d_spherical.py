import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import sam_wrapper as sw

mask_generator = sw.setup()

# Loading the las file from the disk
las = laspy.read("data/ITC_BUILDING.las")

# Transforming to a numpy array
coords = np.vstack((las.x, las.y, las.z))
point_cloud = coords.transpose()

# Gathering the colors
r=(las.red/65535*255).astype(int)
g=(las.green/65535*255).astype(int)
b=(las.blue/65535*255).astype(int)
colors = np.vstack((r,g,b)).transpose()
resolution = 500

# Defining the position in the point cloud to generate a panorama
center_coordinates = [189, 60, 2]

# Function Execution
spherical_image, mapping = sw.generate_spherical_image(center_coordinates, point_cloud, colors, resolution)

# Plotting with matplotlib
fig = plt.figure(figsize=(np.shape(spherical_image)[1]/72,
np.shape(spherical_image)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(spherical_image)
plt.axis('off')

# Saving to the disk
plt.savefig("output/ITC_BUILDING_spherical_projection.jpg")

image_bgr = cv2.imread("output/ITC_BUILDING_spherical_projection.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#t0 = time.time()
result = mask_generator.generate(image_rgb)
#t1 = time.time()

fig = plt.figure(figsize=(np.shape(image_rgb)[1]/72,
np.shape(image_rgb)[0]/72))
fig.add_axes([0,0,1,1])
plt.imshow(image_rgb)
color_mask = sam_masks(result)
plt.axis('off')
plt.savefig("output/ITC_BUILDING_spherical_projection_segmented.jpg")
