

import logging
import json
import os
import numpy
from tqdm import tqdm
import datetime
from torch import nn
from torch.optim import Adam
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchsummary import summary as summary_

from compare_models_speed.ATLAS_models.lib.loader import CustomDataset_3d_gray_out_1
from c_ATLAS.c_2d_3d_swin_hardnet.logger import get_logger
from compare_models_speed.ATLAS_models.lib.loss import DiceCE
# from c_ATLAS.c_2d_3d_swin_hardnet.hardnet_2d_3d import swin_hardnet_2d_3d
from compare_models_speed.ATLAS_models.lib.c_metrics import get_score_from_all_slices_cherhoo_depth
from compare_models_speed.ATLAS_models.lib.swin_hardnet import swin_hardnet_all_transformer_embed_dim_3d


import logging
logging.basicConfig(format='%(asctime)s-<%(funcName)s>-[line:%(lineno)d]-%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # define logging print level
logger.setLevel(logging.INFO)  # define logging print level


# import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # Set the GPUs 2 and 3 to use
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPUs 2 and 3 to use












IMG_H = 224
IMG_W = 224
img_depth = 4
img_depth_out_num = 2


def get_3d_img_list(path, depth):
    names_3d_list = []
    img_depth = depth
    for name in os.listdir(path):
        name_tmps = name.split('_')
        name_id = name.split('_')[3].split(".")[0]
        name_id = int(name_id)

        num_i = name_id
        # if num_i > 30 and name_id <= 142:  ##  전체 MRI scan 이미지 개수  = 112, 앞에 30장 + 뒤 47장  제거
        if num_i <= 189 - img_depth + 1:
            names_3d = []
            for d in range(img_depth):
                name_3d = name_tmps[0] + "_" + name_tmps[1] + "_" + name_tmps[2] + "_{:03d}".format(name_id + d) + ".png"
                names_3d.append(os.path.join(path, name_3d))
            names_3d_list.append(names_3d)
        #break
    return names_3d_list



import numpy as np

################################################













def run(model_path, img_count):
    batch_size = 8*2 # *2
    n_classes = 2
    learning_rate = 0.001
    model_name = "c_hardnet_swin_2d_3d_depth_3"  ##  c_swin_cnn_unet
    patience_scheduler = 5
    patience_early_stop = 10
    ReduceLR_factor = 0.2

    start_iter = 0
    num_epochs = 500

    ## save model log and check points
    logging.info("Let the games begin")
    logging.info("========================================")
    logging.info("# parameters: ")
    logging.info("IMG_H={}".format(IMG_H))
    logging.info("IMG_W={}".format(IMG_W))
    logging.info("img_depth={}".format(img_depth))
    logging.info("img_depth_out_num={}".format(img_depth_out_num))
    logging.info("batch_size={}".format(batch_size))
    logging.info("n_classes={}".format(n_classes))
    logging.info("learning_rate={}".format(learning_rate))
    logging.info("patience_scheduler={}".format(patience_scheduler))
    logging.info("patience_early_stop={}".format(patience_early_stop))
    logging.info("ReduceLR_factor={}".format(ReduceLR_factor))
    # logging.info("logdir={}".format(logdir))
    logging.info("========================================")

    print_interval = 500
    val_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==========================================")
    print('Device:', device)
    print('Current cuda device:', torch.cuda.current_device())
    print("torch.cuda.device_count()=", torch.cuda.device_count())
    print("==========================================")
    device_ids = [0]
    torch.backends.cudnn.benchmark = True

    #### load dataset : by_id
    # val_img_path  = 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/'
    # val_mask_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/masks/'
    # val_data_list = get_3d_img_list(val_img_path, img_depth)
    # val_mask_list = get_3d_img_list(val_mask_path, img_depth)
    # logger.info("----------------------------------------- load val ({}): {}".format(len(val_mask_list), val_img_path))

    train_img_path  = 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/'
    train_mask_path = 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/masks/'
    train_data_list = get_3d_img_list(train_img_path, img_depth)
    train_mask_list = get_3d_img_list(train_mask_path, img_depth)
    logger.info("----------------------------------------- load train ({}): {}".format(len(train_data_list), train_img_path))


    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list[:img_count], train_mask_list[:img_count], img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)
    # val_data   = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = DiceCE()
    criterion.to(device)

    ## create model
    model = swin_hardnet_all_transformer_embed_dim_3d(n_classes=2, in_channels=img_depth, st_img_size=112, st_windows=7).cuda()

    summary_(model, (4, 224, 224), batch_size=batch_size)

    ##  load pre-trained model
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list[:100], train_mask_list[:100], img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)
    start = time.time()
    start_2 = datetime.datetime.now()
    for i, pack in enumerate(train_data):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
    end = time.time()
    end_2 = datetime.datetime.now()
    runing_time = end - start
    runing_time_2 = end_2 - start_2
    print("predict 100 imgs : {}".format(runing_time))
    print("predict 100 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(100 / runing_time))
    print("------------------------------------------------")

    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list[:1000], train_mask_list[:1000], img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)
    start = time.time()
    start_2 = datetime.datetime.now()
    for i, pack in enumerate(train_data):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
    end = time.time()
    end_2 = datetime.datetime.now()
    runing_time = end - start
    runing_time_2 = end_2 - start_2
    print("predict 1,000 imgs : {}".format(runing_time))
    print("predict 1,000 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(1000 / runing_time))
    print("------------------------------------------------")

    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list[:10000], train_mask_list[:10000], img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)
    start = time.time()
    start_2 = datetime.datetime.now()
    for i, pack in enumerate(train_data):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
    end = time.time()
    end_2 = datetime.datetime.now()
    runing_time = end - start
    runing_time_2 = end_2 - start_2
    print("predict 10,000 imgs : {}".format(runing_time))
    print("predict 10,000 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(10000/runing_time))
    print("------------------------------------------------")

    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list[:30000], train_mask_list[:30000], img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)
    start = time.time()
    start_2 = datetime.datetime.now()
    for i, pack in enumerate(train_data):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
    end = time.time()
    end_2 = datetime.datetime.now()
    runing_time = end - start
    runing_time_2 = end_2 - start_2
    print("predict 30,000 imgs : {}".format(runing_time))
    print("predict 30,000 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(30000 / runing_time))
    print("------------------------------------------------")

    return 0



import time
import datetime
if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_score_list = []
    start = time.time()
    start_2 = datetime.datetime.now()
    model_path = "C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/models/STHarDNet_in3d_out2d"
    model_path = os.path.join(model_path, "c_swin_hardnet_3d_224_l3_s3_ATLAS_best_model_0_5457.pkl")
    # a = 1
    # for i in range(1000):
    #     for ii in range(1000):
    #         a = a+1
    img_count = 100
    run(model_path, img_count)

    end = time.time()
    end_2 = datetime.datetime.now()

    print("all:{}".format(end-start))
    print("all:{}".format(end_2 - start_2))



# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/STHarDNet_in3d_out2d.py
# 2021-10-19 18:00:23,330-<run>-[line:183]-INFO: Let the games begin
# 2021-10-19 18:00:23,330-<run>-[line:184]-INFO: ========================================
# 2021-10-19 18:00:23,330-<run>-[line:185]-INFO: # parameters:
# 2021-10-19 18:00:23,330-<run>-[line:186]-INFO: IMG_H=224
# 2021-10-19 18:00:23,330-<run>-[line:187]-INFO: IMG_W=224
# 2021-10-19 18:00:23,330-<run>-[line:188]-INFO: img_depth=4
# 2021-10-19 18:00:23,330-<run>-[line:189]-INFO: img_depth_out_num=2
# 2021-10-19 18:00:23,330-<run>-[line:190]-INFO: batch_size=16
# 2021-10-19 18:00:23,330-<run>-[line:191]-INFO: n_classes=2
# 2021-10-19 18:00:23,330-<run>-[line:192]-INFO: learning_rate=0.001
# 2021-10-19 18:00:23,330-<run>-[line:193]-INFO: patience_scheduler=5
# 2021-10-19 18:00:23,330-<run>-[line:194]-INFO: patience_early_stop=10
# 2021-10-19 18:00:23,330-<run>-[line:195]-INFO: ReduceLR_factor=0.2
# 2021-10-19 18:00:23,330-<run>-[line:197]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# SwinTransformerSys expand initial----depths:[2, 2, 2];depths_decoder:[1, 2, 2, 2];drop_path_rate:0.1;num_classes:6
# 2021-10-19 18:00:24,234-<run>-[line:222]-INFO: ----------------------------------------- load train (32922): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# Parameters: 10400370
# predict 100 imgs : 5.162651777267456
# predict 100 imgs : 0:00:05.162652
# predict FPS is      : 19.36989057451577/s
# ------------------------------------------------
# predict 1,000 imgs : 5.516050577163696
# predict 1,000 imgs : 0:00:05.516050
# predict FPS is      : 181.28912815628877/s
# ------------------------------------------------
# predict 10,000 imgs : 23.047111988067627
# predict 10,000 imgs : 0:00:23.047112
# predict FPS is      : 433.8938434098547/s
# ------------------------------------------------
# predict 30,000 imgs : 64.10534167289734
# predict 30,000 imgs : 0:01:04.105342
# predict FPS is      : 467.9797223931418/s
# ------------------------------------------------
# all:100.22312450408936
# all:0:01:40.223125
#
# Process finished with exit code 0
