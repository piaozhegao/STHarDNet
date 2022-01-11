

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

from compare_models_speed.ATLAS_models.lib.loader import CustomDataset2_gray
from c_ATLAS.c_2d_3d_swin_hardnet.logger import get_logger
from compare_models_speed.ATLAS_models.lib.loss import DiceCE
from compare_models_speed.ATLAS_models.lib.unet_model import UNet


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
img_depth = 1  #4
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



def get_logger(logdir):
    logger = logging.getLogger("train_log")
    ts = str(datetime.datetime.now()).split(".")[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-", "_")
    file_path = os.path.join(logdir, "Speed_compare_run_{}.log".format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
import numpy as np

################################################



from torchsummary import summary as summary_
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

    logdir_tmp = model_path.split("\\")[-1]
    logdir = model_path.replace(logdir_tmp, "")
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    ## save model log and check points
    logger.info("Let the games begin")
    logger.info("========================================")
    logger.info("# parameters: ")
    logger.info("IMG_H={}".format(IMG_H))
    logger.info("IMG_W={}".format(IMG_W))
    logger.info("img_depth={}".format(img_depth))
    logger.info("img_depth_out_num={}".format(img_depth_out_num))
    logger.info("batch_size={}".format(batch_size))
    logger.info("n_classes={}".format(n_classes))
    logger.info("learning_rate={}".format(learning_rate))
    logger.info("patience_scheduler={}".format(patience_scheduler))
    logger.info("patience_early_stop={}".format(patience_early_stop))
    logger.info("ReduceLR_factor={}".format(ReduceLR_factor))
    # logging.info("logdir={}".format(logdir))
    logger.info("========================================")

    print_interval = 500
    val_interval = 500

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("==========================================")
    logger.info('Device:{}'.format(device))
    logger.info('Current cuda device:{}'.format(torch.cuda.current_device()))
    logger.info("torch.cuda.device_count()={}".format(torch.cuda.device_count()))
    logger.info("==========================================")
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
    train_data_list = [train_img_path + i for i in os.listdir(train_img_path)]
    train_mask_list = [train_mask_path + i for i in os.listdir(train_mask_path)]
    # val_data_list = [val_img_path + i for i in os.listdir(val_img_path)]
    # val_mask_list = [val_mask_path + i for i in os.listdir(val_mask_path)]
    train_data_list.extend(train_data_list)
    train_data_list.extend(train_data_list)
    train_mask_list.extend(train_mask_list)
    train_mask_list.extend(train_mask_list)
    logger.info("----------------------------------------- load train ({}): {}".format(len(train_data_list), train_img_path))


    train_data = DataLoader(CustomDataset2_gray(train_data_list[:img_count], train_mask_list[:img_count]), batch_size=batch_size, shuffle=True, num_workers=4)
    # val_data   = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = DiceCE()
    criterion.to(device)

    ## create model
    model = UNet(n_classes=2, n_channels=1).cuda()

    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Parameters:{}'.format(total_params))


    ##  load pre-trained model
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)

    summary_(model, (1, 224, 224), batch_size=batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Parameters:{}'.format(total_params))

    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_data = DataLoader(CustomDataset2_gray(train_data_list[:100], train_mask_list[:100]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 100 imgs : {}".format(runing_time))
    logger.info("predict 100 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(100 / runing_time))
    logger.info("------------------------------------------------")

    train_data = DataLoader(CustomDataset2_gray(train_data_list[:1000], train_mask_list[:1000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 1,000 imgs : {}".format(runing_time))
    logger.info("predict 1,000 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(1000 / runing_time))
    logger.info("------------------------------------------------")

    train_data = DataLoader(CustomDataset2_gray(train_data_list[:10000], train_mask_list[:10000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 10,000 imgs : {}".format(runing_time))
    logger.info("predict 10,000 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(10000/runing_time))
    logger.info("------------------------------------------------")

    train_data = DataLoader(CustomDataset2_gray(train_data_list[:30000], train_mask_list[:30000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 30,000 imgs : {}".format(runing_time))
    logger.info("predict 30,000 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(30000 / runing_time))
    logger.info("------------------------------------------------")

    #######################################
    train_data = DataLoader(CustomDataset2_gray(train_data_list[:50000], train_mask_list[:50000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 50,000 imgs : {}".format(runing_time))
    logger.info("predict 50,000 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(50000 / runing_time))
    logger.info("------------------------------------------------")

    train_data = DataLoader(CustomDataset2_gray(train_data_list[:100000], train_mask_list[:100000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    logger.info("predict 100,000 imgs : {}".format(runing_time))
    logger.info("predict 100,000 imgs : {}".format(runing_time_2))
    logger.info("predict FPS is      : {}/s".format(100000 / runing_time))
    logger.info("------------------------------------------------")

    return 0



import time
import datetime
if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_score_list = []
    start = time.time()
    start_2 = datetime.datetime.now()
    model_path = r"C:\Users\user\Documents\python-code\11_c_paper\compare_models_speed\ATLAS_models\models\UNet_in2d_out2d"
    model_path = os.path.join(model_path, "UNet_224_best_model.pkl")
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









#
# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/UNet_in2d_out2d.py
# 2021-10-20 14:20:54,755-<run>-[line:91]-INFO: Let the games begin
# 2021-10-20 14:20:54,755-<run>-[line:92]-INFO: ========================================
# 2021-10-20 14:20:54,755-<run>-[line:93]-INFO: # parameters:
# 2021-10-20 14:20:54,755-<run>-[line:94]-INFO: IMG_H=224
# 2021-10-20 14:20:54,755-<run>-[line:95]-INFO: IMG_W=224
# 2021-10-20 14:20:54,755-<run>-[line:96]-INFO: img_depth=1
# 2021-10-20 14:20:54,755-<run>-[line:97]-INFO: img_depth_out_num=2
# 2021-10-20 14:20:54,755-<run>-[line:98]-INFO: batch_size=16
# 2021-10-20 14:20:54,755-<run>-[line:99]-INFO: n_classes=2
# 2021-10-20 14:20:54,755-<run>-[line:100]-INFO: learning_rate=0.001
# 2021-10-20 14:20:54,755-<run>-[line:101]-INFO: patience_scheduler=5
# 2021-10-20 14:20:54,755-<run>-[line:102]-INFO: patience_early_stop=10
# 2021-10-20 14:20:54,756-<run>-[line:103]-INFO: ReduceLR_factor=0.2
# 2021-10-20 14:20:54,756-<run>-[line:105]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# 2021-10-20 14:20:54,958-<run>-[line:132]-INFO: ----------------------------------------- load train (33453): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# Parameters: 17266306
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [16, 64, 224, 224]             640
#        BatchNorm2d-2         [16, 64, 224, 224]             128
#               ReLU-3         [16, 64, 224, 224]               0
#             Conv2d-4         [16, 64, 224, 224]          36,928
#        BatchNorm2d-5         [16, 64, 224, 224]             128
#               ReLU-6         [16, 64, 224, 224]               0
#         DoubleConv-7         [16, 64, 224, 224]               0
#          MaxPool2d-8         [16, 64, 112, 112]               0
#             Conv2d-9        [16, 128, 112, 112]          73,856
#       BatchNorm2d-10        [16, 128, 112, 112]             256
#              ReLU-11        [16, 128, 112, 112]               0
#            Conv2d-12        [16, 128, 112, 112]         147,584
#       BatchNorm2d-13        [16, 128, 112, 112]             256
#              ReLU-14        [16, 128, 112, 112]               0
#        DoubleConv-15        [16, 128, 112, 112]               0
#              Down-16        [16, 128, 112, 112]               0
#         MaxPool2d-17          [16, 128, 56, 56]               0
#            Conv2d-18          [16, 256, 56, 56]         295,168
#       BatchNorm2d-19          [16, 256, 56, 56]             512
#              ReLU-20          [16, 256, 56, 56]               0
#            Conv2d-21          [16, 256, 56, 56]         590,080
#       BatchNorm2d-22          [16, 256, 56, 56]             512
#              ReLU-23          [16, 256, 56, 56]               0
#        DoubleConv-24          [16, 256, 56, 56]               0
#              Down-25          [16, 256, 56, 56]               0
#         MaxPool2d-26          [16, 256, 28, 28]               0
#            Conv2d-27          [16, 512, 28, 28]       1,180,160
#       BatchNorm2d-28          [16, 512, 28, 28]           1,024
#              ReLU-29          [16, 512, 28, 28]               0
#            Conv2d-30          [16, 512, 28, 28]       2,359,808
#       BatchNorm2d-31          [16, 512, 28, 28]           1,024
#              ReLU-32          [16, 512, 28, 28]               0
#        DoubleConv-33          [16, 512, 28, 28]               0
#              Down-34          [16, 512, 28, 28]               0
#         MaxPool2d-35          [16, 512, 14, 14]               0
#            Conv2d-36          [16, 512, 14, 14]       2,359,808
#       BatchNorm2d-37          [16, 512, 14, 14]           1,024
#              ReLU-38          [16, 512, 14, 14]               0
#            Conv2d-39          [16, 512, 14, 14]       2,359,808
#       BatchNorm2d-40          [16, 512, 14, 14]           1,024
#              ReLU-41          [16, 512, 14, 14]               0
#        DoubleConv-42          [16, 512, 14, 14]               0
#              Down-43          [16, 512, 14, 14]               0
#          Upsample-44          [16, 512, 28, 28]               0
#            Conv2d-45          [16, 512, 28, 28]       4,719,104
#       BatchNorm2d-46          [16, 512, 28, 28]           1,024
#              ReLU-47          [16, 512, 28, 28]               0
#            Conv2d-48          [16, 256, 28, 28]       1,179,904
#       BatchNorm2d-49          [16, 256, 28, 28]             512
#              ReLU-50          [16, 256, 28, 28]               0
#        DoubleConv-51          [16, 256, 28, 28]               0
#                Up-52          [16, 256, 28, 28]               0
#          Upsample-53          [16, 256, 56, 56]               0
#            Conv2d-54          [16, 256, 56, 56]       1,179,904
#       BatchNorm2d-55          [16, 256, 56, 56]             512
#              ReLU-56          [16, 256, 56, 56]               0
#            Conv2d-57          [16, 128, 56, 56]         295,040
#       BatchNorm2d-58          [16, 128, 56, 56]             256
#              ReLU-59          [16, 128, 56, 56]               0
#        DoubleConv-60          [16, 128, 56, 56]               0
#                Up-61          [16, 128, 56, 56]               0
#          Upsample-62        [16, 128, 112, 112]               0
#            Conv2d-63        [16, 128, 112, 112]         295,040
#       BatchNorm2d-64        [16, 128, 112, 112]             256
#              ReLU-65        [16, 128, 112, 112]               0
#            Conv2d-66         [16, 64, 112, 112]          73,792
#       BatchNorm2d-67         [16, 64, 112, 112]             128
#              ReLU-68         [16, 64, 112, 112]               0
#        DoubleConv-69         [16, 64, 112, 112]               0
#                Up-70         [16, 64, 112, 112]               0
#          Upsample-71         [16, 64, 224, 224]               0
#            Conv2d-72         [16, 64, 224, 224]          73,792
#       BatchNorm2d-73         [16, 64, 224, 224]             128
#              ReLU-74         [16, 64, 224, 224]               0
#            Conv2d-75         [16, 64, 224, 224]          36,928
#       BatchNorm2d-76         [16, 64, 224, 224]             128
#              ReLU-77         [16, 64, 224, 224]               0
#        DoubleConv-78         [16, 64, 224, 224]               0
#                Up-79         [16, 64, 224, 224]               0
#            Conv2d-80          [16, 2, 224, 224]             130
#           OutConv-81          [16, 2, 224, 224]               0
# ================================================================
# Total params: 17,266,306
# Trainable params: 17,266,306
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.06
# Forward/backward pass size (MB): 11551.75
# Params size (MB): 65.87
# Estimated Total Size (MB): 11620.68
# ----------------------------------------------------------------
# Parameters: 17266306
# predict 100 imgs : 4.63908576965332
# predict 100 imgs : 0:00:04.639086
# predict FPS is      : 21.555971362752583/s
# ------------------------------------------------
# predict 1,000 imgs : 7.180324077606201
# predict 1,000 imgs : 0:00:07.180324
# predict FPS is      : 139.2694799276223/s
# ------------------------------------------------
# predict 10,000 imgs : 47.385873794555664
# predict 10,000 imgs : 0:00:47.385874
# predict FPS is      : 211.03335655169317/s
# ------------------------------------------------
# predict 30,000 imgs : 140.71001362800598
# predict 30,000 imgs : 0:02:20.710013
# predict FPS is      : 213.20444243087616/s
# ------------------------------------------------
# all:203.1581106185913
# all:0:03:23.158111
#
# Process finished with exit code 0






