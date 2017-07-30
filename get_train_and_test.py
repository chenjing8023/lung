import random
import glob
import os
import shutil
datadir = "D:\\chenjing\\lung\\dest\\"
train_dest = "D:\\chenjing\\lung\\train_data\\"
test_dest = "D:\\chenjing\\lung\\test_data\\"
if(not os.path.exists(train_dest)):
    os.makedirs(train_dest)
if(not os.path.exists(test_dest)):
    os.makedirs(test_dest)
files = glob.glob(datadir+"*.bin")
random.shuffle(files)
length = int(len(files)/10) + 1
test = files[:length]
train = files[length:]
for sourceFile in test:
    filename = sourceFile.split("\\")[-1]
    targetFile = test_dest + filename
    shutil.copy(sourceFile,  targetFile)
for sourceFile in train:
    filename = sourceFile.split("\\")[-1]
    targetFile = train_dest + filename
    shutil.copy(sourceFile,  targetFile)