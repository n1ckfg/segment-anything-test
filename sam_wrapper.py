# SETUP
import numpy as np
import matplotlib.pyplot as plt
import cv2
import laspy

import torch
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator

def setup(model_name="vit_b", reduce_memory=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_version = None
    
    if (model_name == "vit_h"):
        model_version = "4b8939"
    elif (model_name == "vit_l"):
        model_version = "0b3195"
    elif (model_name == "vit_b"):
        model_version = "01ec64"

    url = "model/sam_" + model_name + "_" + model_version + ".pth"
    print("Using model: \"" + url + "\"")

    sam = sam_model_registry[model_name](checkpoint = url)
    sam.to(device = device)
    mask_generator = None
    if (reduce_memory == True):
        mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=16)
    else:
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

def cloud_to_ortho_image(point_cloud, colors, resolution):
    minx = np.min(point_cloud[:, 0])
    maxx = np.max(point_cloud[:, 0])
    miny = np.min(point_cloud[:, 1])
    maxy = np.max(point_cloud[:, 1])
    width = int((maxx - minx) / resolution) + 1
    height = int((maxy - miny) / resolution) + 1
    image = np.zeros((height, width, 3), dtype=np.uint8)

    for i, point in enumerate(point_cloud):
        x, y, *_ = point
        r, g, b = colors[i]
        pixel_x = int((x - minx) / resolution)
        pixel_y = int((maxy - y) / resolution)
        image[pixel_y, pixel_x] = [r, g, b]

    return image

def generate_spherical_image(point_cloud, colors, resolution_y=500, center_coordinates=None):
    if not center_coordinates:
        pos = np.average(point_cloud, axis=0)
        center_coordinates = [pos[0], pos[1], pos[2]]

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
            try:
                image[iy, ix] = colors[i]
            except:
                pass

    return image, mapping

def remap(value, min1, max1, min2, max2):
    return np.interp(value,[min1, max1],[min2, max2])

def import_point_cloud(url, overrideColors=False):
    las = laspy.read(url)

    coords = np.vstack((las.x, las.y, las.z))
    point_cloud = coords.transpose()

    colors = None
    readColorsFailed = False
    r = None
    g = None
    b = None
    intensity = None

    try:
        if (las.red[0].dtype == "uint16"):
            r = (las.red/65535*255).astype(int)
            g = (las.green/65535*255).astype(int)
            b = (las.blue/65535*255).astype(int)
        else:
            r = (las.red).astype(int)
            g = (las.green).astype(int)
            b = (las.blue).astype(int)
        print("Found color data.")
    except:
        print("No color data found, checking intensity.")
        readColorsFailed = True

    if (readColorsFailed == True or overrideColors == True):
        try:
            if (las.intensity[0].dtype == "uint16"):
                intensity = (las.intensity/65535*255).astype(int)
            else:
                intensity = (las.intensity).astype(int)
            print("Found intensity data.")
        except:
            print("No intensity data found, using elevation as intensity.")
            max_elevation = np.max(las.z)
            min_elevation = np.min(las.z)
            intensity = remap(las.z, min_elevation, max_elevation, 0, 255).astype(int)            

        r = intensity
        g = intensity
        b = intensity

    try:
        if (intensity == None):
            print("Used color.")
    except:
        print("Used intensity.")
    colors = np.vstack((r,g,b)).transpose()

    print("Loaded: \"" + url + "\"")
    return point_cloud, colors

def color_point_cloud(image, point_cloud, mapping):
    image = cv2.resize(image, (len(mapping[0]), len(mapping)))
    h, w = image.shape[:2]
    modified_point_cloud = np.zeros((point_cloud.shape[0], point_cloud.shape[1]+3), dtype=np.float32)
    modified_point_cloud[:, :3] = point_cloud

    for iy in range(h):
        for ix in range(w):
            point_index = mapping[iy, ix]
            if point_index != -1:
                color = image[iy, ix]
                modified_point_cloud[point_index, 3:] = color

    return modified_point_cloud

def export_point_cloud(url, point_cloud):
    # 1. Create a new header
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="random", type=np.int32))

    # 2. Create a Las
    las_o = laspy.LasData(header)
    las_o.x = point_cloud[:,0]
    las_o.y = point_cloud[:,1]
    las_o.z = point_cloud[:,2]
    las_o.red = point_cloud[:,3]
    las_o.green = point_cloud[:,4]
    las_o.blue = point_cloud[:,5]
    las_o.write(url)
    print("Point cloud export successful at: \"" + url + "\"")

    return

def plot_image(image, result, saveUrl=None):
    fig = plt.figure(figsize=(np.shape(image)[1]/72, np.shape(image)[0]/72))
    fig.add_axes([0,0,1,1])
    plt.imshow(image)
    color_mask = sam_masks(result)
    plt.axis("off")
    if (saveUrl != None):
        plt.savefig(saveUrl)
    print("Saved plot image: \"" + saveUrl + "\"")


