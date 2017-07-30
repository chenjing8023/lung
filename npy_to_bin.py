import tensorflow as tf
import glob
import numpy as np
import os
folder = "D:\\chenjing\\lung\\test\\*.npy"
origin = '/Users/chenjing/PycharmProjects/mytask/test/'
dest = 'D:\\chenjing\\lung\\dest\\'
if (not os.path.exists(dest)):
    os.makedirs(dest)


directory = glob.glob(folder)
for file in directory:
    print(file.split('\\'))
    filename = file.split('\\')[-1].split('.npy')[0]
    s = file.split('\\')[-2]
    data = np.load(file)
    result = np.array([1,0])
    for i in data:
        result = np.append(result, i)
    result = np.array(result, dtype=np.uint8)
    result = np.array(result).tostring()
    print (len(result))
    f = open(dest + filename + '.bin', 'wb')
    print(dest + filename + '.bin')
    f.write(result)
    f.close()
