import os
import random
import numpy as np

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR,LambdaLR
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from models.CC import CrowdCounter
from config import cfg
from loading_data import loading_data
from misc.utils import *
from misc.timer import Timer
import pdb

exp_name = cfg.TRAIN.EXP_NAME

if not os.path.exists(cfg.TRAIN.EXP_PATH):
    os.mkdir(cfg.TRAIN.EXP_PATH)


writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)
log_txt = cfg.TRAIN.EXP_PATH + '/' + exp_name + '/' + exp_name + '.txt'

pil_to_tensor = standard_transforms.ToTensor()

train_record = {'best_mae': 1e20, 'mse':1e20,'corr_loss': 0, 'corr_epoch': -1, 'best_model_name': ''}

train_set, train_loader, val_set, val_loader, restore_transform = loading_data()

_t = {'iter time' : Timer(),'train time' : Timer(),'val time' : Timer()} 

rand_seed = cfg.TRAIN.SEED    
if rand_seed is not None:
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

def main():

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True

    net = CrowdCounter().cuda()  

    if cfg.TRAIN.PRE_GCC:
        net.load_state_dict(torch.load(cfg.TRAIN.PRE_GCC_MODEL))
            
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    
    i_tb = 0
    # validate(val_loader, net, -1, restore_transform)
    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        if epoch > cfg.TRAIN.LR_DECAY_START:
            scheduler.step()
            
        # training    
        _t['train time'].tic()
        i_tb = train(train_loader, net, optimizer, epoch, i_tb)
        _t['train time'].toc(average=False)

        print 'train time: {:.2f}s'.format(_t['train time'].diff)
        print '='*20

        # validation
        if epoch%cfg.VAL.FREQ==0 or epoch>cfg.VAL.DENSE_START:
            _t['val time'].tic()
            validate(val_loader, net, epoch, restore_transform)
            _t['val time'].toc(average=False)
            print 'val time: {:.2f}s'.format(_t['val time'].diff)


def train(train_loader, net, optimizer, epoch, i_tb):
    
    for i, data in enumerate(train_loader, 0):
        _t['iter time'].tic()
        img, gt_map, gt_cnt = data
        img = Variable(img).cuda()
        gt_map = Variable(gt_map).cuda()

        optimizer.zero_grad()
        pred_map = net(img, gt_map)
        loss = net.loss
        loss.backward()
        optimizer.step()
        pred_map = pred_map/100.

        if (i + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            

            i_tb = i_tb + 1
            writer.add_scalar('train_loss', loss.data[0], i_tb)

            _t['iter time'].toc(average=False)
            print '[ep %d][it %d][loss %.4f][lr %.4f][%.2fs]' % \
                    (epoch + 1, i + 1, loss.data[0], optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff)
            print '        [cnt: gt: %.1f pred: %.2f]' % (gt_cnt[0], pred_map[0,:,:].sum().data[0])            
        
    return i_tb



def validate(val_loader, net, epoch, restore):
    net.eval()
    print '='*50
    val_loss = []
    mae = 0.0
    mse = 0.0

    for vi, data in enumerate(val_loader, 0):
        img, gt_map, gt_count = data
        img = Variable(img, volatile=True).cuda()
        gt_map = Variable(gt_map, volatile=True).cuda()
        gt_count = gt_count.numpy()

        pred_map = net(img, gt_map)
        val_loss.append(net.loss.data)

        pred_map = pred_map/100.
        pred_map = pred_map.data.cpu().numpy()
        gt_map = gt_map/100.
        gt_map = gt_map.data.cpu().numpy()
        
        
        for i_img in range(pred_map.shape[0]):

            pred_cnt_tmp = np.sum(pred_map[i_img])
            gt_count_tmp = gt_count[i_img]

            mae += abs(gt_count_tmp-pred_cnt_tmp)
            mse += ((gt_count_tmp-pred_cnt_tmp)*(gt_count_tmp-pred_cnt_tmp))
        
        x = []
        if vi==0:
            for idx, tensor in enumerate(zip(img.cpu().data, pred_map, gt_map)):
                if idx>cfg.VIS.VISIBLE_NUM_IMGS:
                    break
                pil_input = restore(tensor[0])
                pil_output = torch.from_numpy(tensor[1]/(tensor[1].max()+1e-10)).repeat(3,1,1)
                pil_label = torch.from_numpy(tensor[2]/(tensor[2].max()+1e-10)).repeat(3,1,1)
                x.extend([pil_to_tensor(pil_input.convert('RGB')), pil_label, pil_output])
            x = torch.stack(x, 0)
            x = vutils.make_grid(x, nrow=3, padding=5)
            writer.add_image(exp_name + '_epoch_' + str(epoch+1), (x.numpy()*255).astype(np.uint8))
        

    mae = mae/val_set.get_num_samples()
    mse = np.sqrt(mse/val_set.get_num_samples())

    loss = np.mean(np.array(val_loss))

    writer.add_scalar('val_loss', loss, epoch + 1)
    writer.add_scalar('mae', mae, epoch + 1)
    writer.add_scalar('mse', mse, epoch + 1)

    snapshot_name = 'ep_%d_mae_%.1f_mse_%.1f' % (epoch + 1, mae, mse)



    if mae < train_record['best_mae']:
        train_record['best_mae'] = mae
        train_record['mse'] = mse
        train_record['corr_epoch'] = epoch + 1
        train_record['corr_loss'] = loss        
        train_record['best_model_name'] = snapshot_name

        with open(log_txt, 'a') as f:
            f.write(snapshot_name + '\n')

        # save model
        to_saved_weight = []


        to_saved_weight = net.state_dict()
        torch.save(to_saved_weight, os.path.join(cfg.TRAIN.EXP_PATH, exp_name, snapshot_name + '.pth'))


    print '='*50
    print exp_name
    print '    '+ '-'*20
    print '    [mae %.2f mse %.2f], [val loss %.4f]' % (mae, mse, loss)         
    print '    '+ '-'*20
    # pdb.set_trace()
    print '[best] [mae %.2f mse %.2f], [loss %.4f], [epoch %d]' % (train_record['best_mae'], train_record['mse'], train_record['corr_loss'], train_record['corr_epoch'])
    print '='*50

    net.train()


if __name__ == '__main__':
    main()








