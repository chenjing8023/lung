import tensorflow as tf
import glob
import numpy as np
import os
folder = '/Users/chenjing/PycharmProjects/mytask/test/*.npy'
origin = '/Users/chenjing/PycharmProjects/mytask/test/'
dest = '/Users/chenjing/PycharmProjects/mytask/destdata/'
if (not os.path.exists(dest)):
    os.makedirs(dest)


directory = glob.glob(folder)
for file in directory:
    filename = file.split('/')[-1].split('.npy')[0]

    data = np.load(origin + filename + '.npy')
    result = np.array([1])
    print(data.shape)
    for i in data:
        result = np.append(result, i)
    result = np.array(result,dtype=np.uint8)
    print(result.shape)
    result = np.array(result).tostring()
    print (len(result))
    f = open(dest + filename + '.bin', 'wb')
    f.write(data)
    f.close()