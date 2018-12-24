import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()

cfg = __C
__C.DATA = edict()
__C.TRAIN = edict()
__C.VAL = edict()
__C.VIS = edict()

#------------------------------DATA------------------------
__C.DATA.STD_SIZE = (768,1024)
__C.DATA.DATA_PATH = '/media/D/DataSet/CC/UCF-qnrf/768x1024_1221'                  

__C.DATA.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449]) # UCF QNRF
__C.DATA.LABEL_FACTOR = 1
__C.DATA.LOG_PARA = 100.

#------------------------------TRAIN------------------------
__C.TRAIN.INPUT_SIZE = (576,768)
__C.TRAIN.SEED = 3035
__C.TRAIN.BATCH_SIZE = 6

__C.TRAIN.PRE_GCC = False
__C.TRAIN.PRE_GCC_MODEL = './pre/Pretrained_GCC.pth'

__C.TRAIN.GPU_ID = [0,1]

# learning rate settings
__C.TRAIN.LR = 1e-5
__C.TRAIN.LR_DECAY = 0.995
__C.TRAIN.LR_DECAY_START = -1
__C.TRAIN.NUM_EPOCH_LR_DECAY = 1 # epoches

__C.TRAIN.MAX_EPOCH = 1000

# output 
__C.TRAIN.PRINT_FREQ = 20

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.TRAIN.EXP_NAME = now \
                    + '_resSFCN_'\
                    + '_' + str(__C.TRAIN.LR) \
                    + '_GCC' + str(__C.TRAIN.PRE_GCC) 


__C.TRAIN.EXP_PATH = './exp'

#------------------------------VAL------------------------
__C.VAL.BATCH_SIZE = 2 # imgs
__C.VAL.DENSE_START = 50
__C.VAL.FREQ = 5 # After 300 epoches, the freq is set as 1

#------------------------------VIS------------------------
__C.VIS.VISIBLE_NUM_IMGS = 2

#------------------------------MISC------------------------


#================================================================================
#================================================================================
#================================================================================  