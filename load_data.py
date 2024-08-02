import os
import cv2
import pickle 
import json 
import math
from scipy.io import savemat
import numpy as np

def load_data(target):
    # dirpath = "./data/{}".format(target)
    dirpath = target
    pickle_loc = "{}/Data".format(dirpath)
    output_loc = "{}/UnzipData".format(dirpath)
    cfg_path = "{}/Config.json".format(dirpath)


    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    for agents in cfg["agents"][0]["sensors"]:
        if agents["sensor_type"] != "ImagingSonar": continue
        hfov = agents["configuration"]["Azimuth"]
        vfov = agents["configuration"]["Elevation"]
        min_range = agents["configuration"]["RangeMin"]
        max_range = agents["configuration"]["RangeMax"]
        hfov = math.radians(hfov)
        vfov = math.radians(vfov)

    # os.makedirs(output_loc)
    images = []
    sensor_poses = []

    ds_factor = cfg["ds_factor"]
    for pkls in os.listdir(pickle_loc):
        filename = "{}/{}".format(pickle_loc, pkls)
        with open(filename, 'rb') as f:
            state = pickle.load(f)
            image = state["ImagingSonar"]
            s = image.shape
            # get rid of preprocessing? 
            if image.dtype == np.uint8:
                # print("hey")
                image = image / 255
            if cfg["simple_denoise"]:
                image[image < 0.2] = 0
                image[s[0]- 200:, :] = 0
            pose = state["PoseSensor"]
            pose[:3, 3] /= ds_factor
            images.append(image)
            sensor_poses.append(pose)

    data = {
        "images": images,
        "images_no_noise": [],
        "sensor_poses": sensor_poses,
        "min_range": min_range / ds_factor,
        "max_range": max_range / ds_factor,
        "hfov": hfov,
        "vfov": vfov
    }
    
    # savemat('{}/{}.mat'.format(dirpath,target), data, oned_as='row')
    return data
