import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import misc.transforms as own_transforms
from datasets.UCF_QNRF import UCF_QNRF
from config import cfg

def loading_data():
	# shanghai Tech A
    mean_std = cfg.DATA.MEAN_STD
    log_para = cfg.DATA.LOG_PARA
    factor =cfg.DATA.LABEL_FACTOR
    train_main_transform = own_transforms.Compose([
    	own_transforms.RandomCrop(cfg.TRAIN.INPUT_SIZE),
    	own_transforms.RandomHorizontallyFlip()
    ])
    val_main_transform = None
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    gt_transform = standard_transforms.Compose([
        own_transforms.GTScaleDown(factor),
        own_transforms.LabelNormalize(log_para)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    train_set = UCF_QNRF(cfg.DATA.DATA_PATH+'/train', 'train',main_transform=train_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)

    val_set = UCF_QNRF(cfg.DATA.DATA_PATH+'/test', 'test', main_transform=val_main_transform, img_transform=img_transform, gt_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=8, shuffle=True, drop_last=False)

    return train_set, train_loader, val_set, val_loader, restore_transform
