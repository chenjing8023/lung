from __future__ import print_function, division
import SimpleITK as sitk
import math
import scipy.ndimage
import numpy as np
import csv
import cv2
import os
from glob import glob
import pandas as pd
import random
from PIL import Image
import scipy
from scipy.ndimage.interpolation import zoom
from skimage import measure, segmentation
import SimpleITK as sitk
import scipy.ndimage as ndimage
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import skimage
from skimage.morphology import convex_hull_image, ball

try:
    from tqdm import tqdm  # long waits are not fun
except:

    tqdm = lambda x: x
workspace = "/data/luna/mytask/data/"
dataspace="/data/unzip/"

###################################################################################
class Sober_luna16(object):
    def __init__(self, workspace):

        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace, "all_patients/")
        self.tmp_workspace = self.workspace
        self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
        self.df_annotations = pd.read_csv(self.workspace + "csv_files/small_nodule.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(
            lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()

    def normalize(self, image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image


    def set_window_width(self, image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):

        image[image > MAX_BOUND] = MAX_BOUND
        image[image < MIN_BOUND] = MIN_BOUND
        return image


    def get_filename(self, file_list, case):
        for f in file_list:
            if case in f:
                return (f)

    def make_mask(self, center, diam, z, width, height, spacing, origin):

        mask = np.zeros([height, width])
        padding = 0

        v_center = (center - origin) / spacing
        v_diam = int(diam / spacing[0] / 2 + 8)
        v_xmin = np.max([0, int(v_center[0] - v_diam) - padding])
        v_xmax = np.min([width - 1, int(v_center[0] + v_diam) + padding])
        v_ymin = np.max([0, int(v_center[1] - v_diam) - padding])
        v_ymax = np.min([height - 1, int(v_center[1] + v_diam) + padding])
        v_xrange = range(v_xmin, v_xmax + 1)
        v_yrange = range(v_ymin, v_ymax + 1)
        # Convert back to world coordinates for distance calculation
        # Fill in 1 within sphere around nodule
        print("mask--" + str(v_xmin) + ":" + str(v_xmax) + "," + str(v_ymin) + ":" + str(v_ymax))
        for v_x in v_xrange:
            for v_y in v_yrange:
                p_x = spacing[0] * v_x + origin[0]
                p_y = spacing[1] * v_y + origin[1]
                if np.linalg.norm(center - np.array([p_x, p_y, z])) <= diam:
                    mask[int((p_y - origin[1]) / spacing[1]), int((p_x - origin[0]) / spacing[0])] = 1.0
        return (mask)

    def rotate(self, image, angle):
        im1 = Image.open(image)
        im2 = im1.rotate(angle)
        return np.array(im2)

    def transpose(self, image, chage):
        im1 = Image.open(image)
        im2 = im1.transpose(chage)
        return np.array(im2)

    def get_mask(self, data, center, diam, z, width, height, spacing, origin):

        my_max = 40
        padding = 0
        mask = np.zeros([my_max, my_max])
        random_mask = np.zeros([my_max, my_max])
        mask_45 = np.zeros([my_max, my_max])
        mask_90 = np.zeros([my_max, my_max])

        v_center = (center - origin) / spacing

        v_diam = int(diam / spacing[0] / 2 + 8)


        start = -1
        if (int(my_max - v_diam * 2) > 2):
            start = -int(my_max - v_diam * 2) + 1


        v_xmin = np.max([0, int(v_center[0] - v_diam) - padding])
        v_xmax = np.min([width - 1, int(v_xmin + my_max)])

        v_ymin = np.max([0, int(v_center[1] - v_diam) - padding])
        v_ymax = np.min([height - 1, int(v_ymin + my_max)])

        v_xrange = range(v_xmin, v_xmax)
        v_yrange = range(v_ymin, v_ymax)
        # Convert back to world coordinates for distance calculation
        # Fill in 1 within sphere around nodule


        x_margin = v_xmax - v_xmin
        y_margin = v_ymax - v_ymin


        x_start = np.max([0, v_xmin + random.randint(start, -1)])
        y_start = np.max([0, v_ymin + random.randint(start, -1)])

        x_end = np.min([width - 1, x_start + my_max])
        y_end = np.min([width - 1, y_start + my_max])
        # print(str(v_xmin)+":"+str(v_xmax)+","+str(v_ymin)+":"+str(v_ymax))
        # print(str(x_start)+":"+str(x_end)+","+str(y_start)+":"+str(y_end))

        mask[0:y_margin, 0:x_margin] = data[v_ymin:v_ymax, v_xmin:v_xmax]  
        random_mask[0:y_end - y_start, 0:x_end - x_start] = data[y_start:y_end, x_start:x_end] 
        '''
        for v_x in v_xrange:
            for v_y in v_yrange:
                if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                    mask[int(v_y),int(v_x)] = data[int(v_y),int(v_x)]
        '''
        return [mask, random_mask]

    def lung_segmentation(self, data_ori):
        marker_internal = data_ori < -400

        bimg = np.zeros(data_ori.shape, dtype=np.bool)
        for i in range(bimg.shape[2]):
            bimg[:, :, i] = segmentation.clear_border(marker_internal[:, :, i])

        ball_kernel = ball(radius=3, dtype=np.float32)
        bimg = ndimage.morphology.binary_closing(bimg, structure=ball_kernel, iterations=3)

        labels = measure.label(bimg)
        probs = measure.regionprops(labels)
        areas = [r.area for r in probs]
        areas.sort()

        lungRegion = np.zeros(bimg.shape, dtype=np.bool)

        if len(areas) >= 2:
            mark = areas[-2] > areas[-1] / 4
            for region in probs:
                if region.area == areas[-1] or (mark and region.area == areas[-2]):
                    for coordinates in region.coords:
                        lungRegion[coordinates[0], coordinates[1], coordinates[2]] = True
        elif len(areas) >= 1:
            for region in probs:
                if region.area == areas[-1]:
                    lungRegion[coordinates[0], coordinates[1], coordinates[2]] = True

        lungRegion = ndimage.morphology.binary_opening(lungRegion, structure=ball_kernel, iterations=1)
        ball_kernel = skimage.morphology.ball(radius=4, dtype=np.float32)
        lungRegion = ndimage.morphology.binary_dilation(lungRegion, structure=ball_kernel, iterations=3)
        return lungRegion

    def myselfhandler(self):
        for fcount, img_file in enumerate(tqdm(self.ls_all_patients)):
            mini_df = self.df_annotations[self.df_annotations["file"] == img_file]  
            # print(img_file[len(img_file)-11:len(img_file)-4])
            if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
                # load the data once
                itk_img = sitk.ReadImage(img_file)
                img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
                segment = self.lung_segmentation(img_array)
                img_array = img_array * segment
                img_array = 255.0 * self.normalize(img_array)
                num_z, height, width = img_array.shape  # height * width constitute the transverse plane
                origin = np.array(itk_img.GetOrigin())  # x,y,z   Origin in world coordinates (mm)
                spacing = np.array(itk_img.GetSpacing())  #


                #for i in range(len(img_array)):
                #    cv2.imwrite(self.tmp_workspace + "img/" + str(i) + ".jpg", img_array[i])
                for node_idx, cur_row in mini_df.iterrows():
                    node_x = cur_row["coordX"]
                    node_y = cur_row["coordY"]
                    node_z = cur_row["coordZ"]
                    diam = cur_row["diameter_mm"]

                    w_nodule_center = np.array([node_x, node_y, node_z])  #
                    v_nodule_center = np.rint(
                        (w_nodule_center - origin) / spacing)  #  (still x,y,z ordering)

                    i_z = int(v_nodule_center[2])
                    nodule_mask = self.make_mask(w_nodule_center, diam, i_z * spacing[2] + origin[2], width, height,
                                                 spacing, origin)
                    # nodule_mask = scipy.ndimage.interpolation.zoom(nodule_mask, [0.5, 0.5], mode='nearest')
                    # nodule_mask[nodule_mask<0.5] = 0
                    # nodule_mask[nodule_mask > 0.5] = 1
                    nodule_mask = nodule_mask.astype('int8')
                    slice = img_array[i_z]
                    # slice = scipy.ndimage.interpolation.zoom(slice, [0.5, 0.5], mode='nearest')
                    #slice = 255.0 * slice
                    slice = slice.astype(np.uint8)  #
                    node = self.get_mask(slice, w_nodule_center, diam, i_z * spacing[2] + origin[2], width, height,
                                         spacing, origin)


                    # ===================================

                    nodule_mask = 255.0 * nodule_mask
                    nodule_mask = nodule_mask.astype(np.uint8)
                    #cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s.jpg"
                    #                         % (fcount, node_idx, i_z, cur_row["seriesuid"])), slice)
                    name = os.path.join(self.tmp_workspace, "jpg_smll/%04d_%04d_%04d_%s_node.jpg"
                                        % (fcount, node_idx, i_z, cur_row["seriesuid"]))
                    cv2.imwrite(name, node[0])
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_node_random.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuid"])), node[1])
                    mask_45 = self.rotate(name, 45)
                    mask_90 = self.rotate(name, 90)
                    flip_left_right = self.transpose(name, Image.FLIP_LEFT_RIGHT)
                    flip_top_bottom = self.transpose(name, Image.FLIP_TOP_BOTTOM)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_node_45.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuid"])), mask_45)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_node_90.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuid"])), mask_90)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_node_left.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuid"])), flip_left_right)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_node_top.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuid"])), flip_top_bottom)
                    cv2.imwrite(os.path.join(self.tmp_workspace, "jpg_small/%04d_%04d_%04d_%s_o.jpg"
                                             % (fcount, node_idx, i_z, cur_row["seriesuids"])), nodule_mask)
                    np.save(os.path.join(self.tmp_workspace, "npy_small/base/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"],fcount, node_idx,i_z)), node[0])
                    np.save(os.path.join(self.tmp_workspace, "npy_small/random/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"], fcount, node_idx, i_z)),node[1])
                    np.save(os.path.join(self.tmp_workspace,"npy_small/base_45/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"], fcount, node_idx, i_z)),mask_45)
                    np.save(os.path.join(self.tmp_workspace,"npy_small/base_90/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"], fcount, node_idx, i_z)),mask_90)
                    np.save(os.path.join(self.tmp_workspace,"npy_small/left/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"], fcount, node_idx, i_z)),flip_left_right)
                    np.save(os.path.join(self.tmp_workspace,"npy_small/top/%s_%04d_%04d_%04d.npy"
                                         % (cur_row["seriesuid"], fcount, node_idx, i_z)),flip_top_bottom)
                    np.save(os.path.join(self.tmp_workspace, "mask/%s_%04d_%04d_%04d_small_o.npy"
                                         % (cur_row["seriesuid"],fcount, node_idx,i_z)),nodule_mask)

if __name__ == '__main__':
    sl = Sober_luna16(workspace)
    sl.myselfhandler()
