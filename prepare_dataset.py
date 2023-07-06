import os
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from dataLoader import CustomDataset
import numpy as np
path = "data/lfw_funneled/"
colorpath = "data/color/"
bwpath = "data/bw/"

cv.namedWindow("im", cv.WINDOW_NORMAL)

if not os.path.exists(colorpath):
    os.mkdir(colorpath)
if not os.path.exists(bwpath):
    os.mkdir(bwpath)


# i = 0
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file

#         if file.endswith((".jpg", ".jpeg", ".png")):
#             print(filepath)
#             im = cv.imread(filepath)
#             cv.imshow("im",im)
#             cv.waitKey(1)
#             cv.imwrite(colorpath+"color_"+str(i)+".jpg",im)
#             imbw = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
#             cv.imwrite(bwpath+"bw_"+str(i)+".jpg",imbw)
#             i+=1

dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_id, (X,y) in enumerate(data_loader):
    X = X[0].numpy()
    y = y[0].numpy()
    #breakpoint()
    #cvImg = cv.cvtColor(X, cv.COLOR_RGB2BGR)
    cv.imshow("im", X)
    cv.imshow("im2", y)
    cv.waitKey(0)
