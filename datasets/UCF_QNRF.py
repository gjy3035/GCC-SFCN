import numpy as np
import os
import random
import pandas as pd
import sys
import torch
from torch.utils import data
from PIL import Image, ImageOps
from config import cfg


class UCF_QNRF(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files) 

        self.mode = mode
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        
        
    
    def __getitem__(self, index):
        fname = self.data_files[index]

        input_size = []
        if self.mode=='train':
        	input_size = cfg.DATA.STD_SIZE
        elif self.mode=='test':
        	input_size = cfg.DATA.STD_SIZE

        # print fname
        img, den = self.read_image_and_gt(fname, input_size)
      
        if self.main_transform is not None:
            img, den = self.main_transform(img,den) 

        if self.img_transform is not None:
            img = self.img_transform(img)

        # den = torch.from_numpy(np.array(den))
        gt_count = torch.from_numpy(np.array(den)).sum() 

        if self.gt_transform is not None:
            den = self.gt_transform(den)      
            
        return img, den, gt_count

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname, input_size):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')
        wd_1, ht_1 = img.size

        # pdb.set_trace()

        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        den = den.astype(np.float32, copy=False)
        # add padding       
        den = Image.fromarray(den)
        if wd_1 < input_size[1]:
            dif = input_size[1] - wd_1
            img = ImageOps.expand(img, border=(0,0,dif,0), fill=0)
            den = ImageOps.expand(den, border=(0,0,dif,0), fill=0)           
            
        if ht_1 < input_size[0]:
            dif = input_size[0] - ht_1
            img = ImageOps.expand(img, border=(0,0,0,dif), fill=0)
            den = ImageOps.expand(den, border=(0,0,0,dif), fill=0)            
        
        return img, den

                


    def get_num_samples(self):
        return self.num_samples       
            
        