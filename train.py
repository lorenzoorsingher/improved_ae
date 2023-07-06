from model_files import model
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from dataLoader import CustomDataset
import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
import os
from copy import copy
import time

LOAD_CHKP = False
VIS_DEBUG = True
SAVE_PATH = "checkpoints/"
VISUAL = 20
BATCH = 16
EPOCH = 300
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))


#folders for checkpoints and degub images
current_savepath = SAVE_PATH + "run_"+str(round(time.time()))+"/"
img_savepath = current_savepath + "imgs/"
os.mkdir(current_savepath)
os.mkdir(img_savepath)

#load the custom dattaset and correspondent dataloader
dataset = CustomDataset()
data_loader = DataLoader(dataset, batch_size=BATCH, shuffle=True)

#load the model
ae = model.SimplerAE2().to(DEVICE)

#print model and parameters number
model_parameters = filter(lambda p: p.requires_grad, ae.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params, " total params")
print(ae)

#setup optimizer and loss function
opt = SGD(ae.parameters(), lr=LR)
lossFunc = nn.MSELoss()

epoch = 0

#if set, load the a saved checkpoint
if (LOAD_CHKP):
    chkp_path = "checkpoints/run_1688585646/checkpoint_299.chkp"
    checkpoint = torch.load(chkp_path)
    ae.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

### load example image (not from dataset)
Xmple = cv.imread("data/bwme.jpg")
ymple = cv.imread("data/colme.jpg")
Xmple = cv.cvtColor(Xmple, cv.COLOR_BGR2GRAY)

Xmple = cv.resize(Xmple, dataset.img_dim)
ymple = cv.resize(ymple, dataset.out_size)

Xmple, ymple = dataset.normalize(Xmple, ymple)

Xmple = torch.tensor([[Xmple]])
ymple = torch.tensor([ymple])

Xmple = Xmple.permute(0,1, 3, 2)
ymple = ymple.permute(0,3, 2, 1)
Xmple = Xmple.float()
ymple = ymple.float()
########################################

cv.namedWindow("encode_decode_result", cv.WINDOW_NORMAL)

#training loop
ae.train()
for i in range(EPOCH):
    print("############## EPOCH n",i,"\n")

    epochLoss = 0
    batchItems = 0
    stop = True
    count = 0

    #loop thru single batch
    for batch_id, (X,y) in enumerate(data_loader):
        
        #convert tensors to float and load em to device
        X = X.float()
        y = y.float()
        (X,y) = (X.to(DEVICE), y.to(DEVICE))
        
        #actual trainign lol #####
        predictions,_ = ae(X)
    
        loss = lossFunc(predictions, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        ##########################

        count+=1
        if count%VISUAL==0 and VIS_DEBUG:
            
            #basically multiply std and add mean for each channel
            Ximg, _ = dataset.denormalize(copy(X[0].detach().transpose(0,2).numpy()),None)
            pred, yimg = dataset.denormalize(copy(predictions[0].detach().transpose(0,2).numpy()),copy(y[0].detach().transpose(0,2).numpy()))
            
            #tensor to ndarray, resize and gray to bgr to allaw hstacking
            Ximg = Ximg.astype(np.uint8)
            Ximg = cv.resize(Ximg, yimg.shape[:2])
            Ximg = cv.cvtColor(Ximg,cv.COLOR_GRAY2BGR)
            yimg = yimg.astype(np.uint8)
            #some values exceed 254 or are negative (no tanh, sigmoid or similar in net 
            #because data is already standardized)
            pred[pred > 254] = 254 
            pred[pred < 0] = 0 
            pred = pred.astype(np.uint8)
            
            #same as above
            example_pred, _ = ae(Xmple)
            ex_Ximg, _ = dataset.denormalize(copy(Xmple[0].detach().transpose(0,2).numpy()),None)
            ex_pred, ex_yimg = dataset.denormalize(copy(example_pred[0].detach().transpose(0,2).numpy()),copy(ymple[0].detach().transpose(0,2).numpy()))

            ex_Ximg = ex_Ximg.astype(np.uint8)
            ex_Ximg = cv.resize(ex_Ximg, yimg.shape[:2])
            ex_Ximg = cv.cvtColor(ex_Ximg,cv.COLOR_GRAY2BGR)
            ex_yimg = ex_yimg.astype(np.uint8)
            ex_pred[ex_pred > 254] = 254 
            ex_pred[ex_pred < 0] = 0 
            ex_pred = ex_pred.astype(np.uint8)

            cv.imshow("encode_decode_result", np.vstack([np.hstack([Ximg,yimg,pred]),np.hstack([ex_Ximg,ex_yimg,ex_pred])]))
            cv.imwrite(img_savepath+"img_"+str(i)+"_"+str(count)+".jpg",np.hstack([ex_Ximg,ex_yimg,ex_pred]))
            print("batch_loss: ", loss.item()/BATCH)
            cv.waitKey(1)

        epochLoss += loss.item()
        batchItems += BATCH

    #save checkpoint
    print("[SAVE] saving checkpoint...")
    torch.save({
            'epoch': i,
            'model_state_dict': ae.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': epochLoss/batchItems,
            }, current_savepath + "checkpoint_"+str(i)+".chkp")
    print("loss: ", epochLoss/batchItems)