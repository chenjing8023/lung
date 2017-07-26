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
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x

workspace = "/home/hadoop/tmp/Luna16/data/"
###################################################################################
class Sober_luna16(object):
    def __init__(self, workspace):
        """param: workspace: all_patients的父目录"""
        self.workspace = workspace
        self.all_patients_path = os.path.join(self.workspace,"all_patients/")
        self.tmp_workspace = os.path.join(self.workspace,"slices_masks/")
        self.ls_all_patients = glob(self.all_patients_path + "*.mhd")
        
        self.df_annotations = pd.read_csv(self.workspace + "csv_files/annotations.csv")
        self.df_annotations["file"] = self.df_annotations["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.df_annotations = self.df_annotations.dropna()
        
        self.candidates = pd.read_csv(self.workspace + "csv_files/candidates.csv")
        self.candidates["file"] = self.candidates["seriesuid"].map(lambda file_name: self.get_filename(self.ls_all_patients, file_name))
        self.candidates = self.candidates.dropna()
    def get_filename(self,file_list, case):
        for f in file_list:
            if case in f:
                return (f)
    def myselfhandler(self):
        """自己处理"""
        nodule=[]
        no_nodule=[]
        small_nodule=[]
        for fcount, img_file in enumerate(tqdm(self.ls_all_patients)):
            mini_df = self.df_annotations[self.df_annotations["file"] == img_file]  # 获取这个病人的所有结节信息
            candidates= self.candidates[self.candidates["file"] == img_file]
            #print(mini_df)
            count=1
            for index,candidate in candidates.iterrows():
                if(candidate['class']==1):
                    flag=0
                    diameter_mm=3.0
                    tmp=''
                    for i,df in mini_df.iterrows():
                        diameter_mm=3.0
                        x= abs(candidate['coordX']-df['coordX'])
                        y=abs(candidate['coordY']-df['coordY'])
                        z=abs(candidate['coordZ']-df['coordZ'])
                        #通过坐标方向小于2来判断是否为同一个结节
                        if(x<3 and y<3 and z<3):
                            flag=1
                            diameter_mm=df['diameter_mm']
                            tmp=df
                            break
                    if(flag==1):
                        data=[tmp['seriesuid'],tmp['coordX'],tmp['coordY'],tmp['coordZ'],diameter_mm,candidate['class']]
                        nodule.append(data)
                    else:
                        data=[candidate['seriesuid'],candidate['coordX'],candidate['coordY'],candidate['coordZ'],diameter_mm,candidate['class']]
                        small_nodule.append(data)
                    
                else:
                    data=data=[candidate['seriesuid'],candidate['coordX'],candidate['coordY'],candidate['coordZ'],0,candidate['class']]
                    no_nodule.append(data)
                count+=1
        columns=['seriesuid','coordX','coordY','coordZ','diameter_mm','class']
        data=[] #合并结节和非结节数组
        for i in nodule:
            data.append(i)
        for i in no_nodule:
            data.append(i)
        for i in small_nodule:
            data.append(i)
        
        data = pd.DataFrame(data,columns=columns)
        data.to_csv('total.csv',index=False)
        #small_nodule = pd.DataFrame(small_nodule,columns=columns)
        #small_nodule.to_csv('small_nodule.csv',index=False)
        #no_nodule = pd.DataFrame(no_nodule,columns=columns)
        #no_nodule.to_csv('no_nodule.csv',index=False)
        #nodule = pd.DataFrame(nodule,columns=columns)
        #nodule.to_csv('nodule.csv',index=False)
if __name__ == '__main__':
    sl = Sober_luna16(workspace)
    sl.myselfhandler()
