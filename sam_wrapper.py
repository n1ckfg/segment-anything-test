# SETUP
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

def setup(model_name="vit_b"):
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  
  model_version = None
  
  if (model_name == "vit_h"):
    model_version = "4b8939"
  elif (model_name == "vit_l"):
    model_version = "0b3195"
  elif (model_name == "vit_b"):
    model_version = "01ec64"

  url = "model/sam_" + model_name + "_" + model_version + ".pth"

  sam = sam_model_registry[model_name](checkpoint = url)
  sam.to(device = device)
  mask_generator = SamAutomaticMaskGenerator(sam)

  return mask_generator

# METHODS
def sam_masks(anns):
  if len(anns) == 0:
    return

  sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
  ax = plt.gca()
  ax.set_autoscale_on(False)
  c_mask=[]

  for ann in sorted_anns:
    m = ann['segmentation']
    img = np.ones((m.shape[0], m.shape[1], 3))
    color_mask = np.random.random((1, 3)).tolist()[0]

    for i in range(3):
      img[:,:,i] = color_mask[i]

    ax.imshow(np.dstack((img, m*0.8)))
    c_mask.append(img)

  return c_mask

def cloud_to_image(pcd_np, resolution):
  minx = np.min(pcd_np[:, 0])
  maxx = np.max(pcd_np[:, 0])
  miny = np.min(pcd_np[:, 1])
  maxy = np.max(pcd_np[:, 1])
  width = int((maxx - minx) / resolution) + 1
  height = int((maxy - miny) / resolution) + 1
  image = np.zeros((height, width, 3), dtype=np.uint8)

  for point in pcd_np:
    x, y, *_ = point
    r, g, b = point[-3:]
    pixel_x = int((x - minx) / resolution)
    pixel_y = int((maxy - y) / resolution)
    image[pixel_y, pixel_x] = [r, g, b]

  return image

def generate_spherical_image(center_coordinates, point_cloud, colors, resolution_y=500):
  # Translate the point cloud by the negation of the center coordinates
  translated_points = point_cloud - center_coordinates

  # Convert 3D point cloud to spherical coordinates
  theta = np.arctan2(translated_points[:, 1], translated_points[:, 0])
  phi = np.arccos(translated_points[:, 2] / np.linalg.norm(translated_points, axis=1))

  # Map spherical coordinates to pixel coordinates
  x = (theta + np.pi) / (2 * np.pi) * (2 * resolution_y)
  y = phi / np.pi * resolution_y

  # Create the spherical image with RGB channels
  resolution_x = 2 * resolution_y
  image = np.zeros((resolution_y, resolution_x, 3), dtype=np.uint8)

  # Create the mapping between point cloud and image coordinates
  mapping = np.full((resolution_y, resolution_x), -1, dtype=int)
  # Assign points to the image pixels
  for i in range(len(translated_points)):
    ix = np.clip(int(x[i]), 0, resolution_x - 1)
    iy = np.clip(int(y[i]), 0, resolution_y - 1)
    if mapping[iy, ix] == -1 or np.linalg.norm(translated_points[i]) < np.linalg.norm(translated_points[mapping[iy, ix]]):
      mapping[iy, ix] = i
      image[iy, ix] = colors[i]

  return image, mapping

def color_point_cloud(image_path, point_cloud, mapping):
  image = cv2.imread(image_path)
  h, w = image.shape[:2]
  modified_point_cloud = np.zeros((point_cloud.shape[0],
  point_cloud.shape[1]+3), dtype=np.float32)
  modified_point_cloud[:, :3] = point_cloud

  for iy in range(h):
    for ix in range(w):
      point_index = mapping[iy, ix]
      if point_index != -1:
        color = image[iy, ix]
        modified_point_cloud[point_index, 3:] = color

  return modified_point_cloud

def export_point_cloud(cloud_path, modified_point_cloud):
  # 1. Create a new header
  header = laspy.LasHeader(point_format=3, version="1.2")
  header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))

  # 2. Create a Las
  las_o = laspy.LasData(header)
  las_o.x = modified_point_cloud[:,0]
  las_o.y = modified_point_cloud[:,1]
  las_o.z = modified_point_cloud[:,2]
  las_o.red = modified_point_cloud[:,3]
  las_o.green = modified_point_cloud[:,4]
  las_o.blue = modified_point_cloud[:,5]
  las_o.write(cloud_path)
  print("Export succesful at: ", cloud_path)

  return

# Example
#export_point_cloud("/content/pcd_results.las", modified_point_cloud)


