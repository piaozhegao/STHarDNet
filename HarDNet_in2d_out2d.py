

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
# from c_ATLAS.c_2d_3d_swin_hardnet.logger import get_logger
from compare_models_speed.ATLAS_models.lib.loss import DiceCE
from compare_models_speed.ATLAS_models.lib.c_hardnet_swin_ende_decoder import hardnet


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
    model = hardnet(n_classes=2, in_channels=1).cuda()

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


    #################################
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
    model_path = r"C:\Users\user\Documents\python-code\11_c_paper\compare_models_speed\ATLAS_models\models\hardnet_in2d_out2d"
    model_path = os.path.join(model_path, "c_hardnet_ATLAS_best_model.pkl")
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
# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/HarDNet_in2d_out2d.py
# 2021-10-20 10:44:17,839-<run>-[line:91]-INFO: Let the games begin
# 2021-10-20 10:44:17,839-<run>-[line:92]-INFO: ========================================
# 2021-10-20 10:44:17,839-<run>-[line:93]-INFO: # parameters:
# 2021-10-20 10:44:17,839-<run>-[line:94]-INFO: IMG_H=224
# 2021-10-20 10:44:17,839-<run>-[line:95]-INFO: IMG_W=224
# 2021-10-20 10:44:17,839-<run>-[line:96]-INFO: img_depth=1
# 2021-10-20 10:44:17,839-<run>-[line:97]-INFO: img_depth_out_num=2
# 2021-10-20 10:44:17,839-<run>-[line:98]-INFO: batch_size=16
# 2021-10-20 10:44:17,839-<run>-[line:99]-INFO: n_classes=2
# 2021-10-20 10:44:17,839-<run>-[line:100]-INFO: learning_rate=0.001
# 2021-10-20 10:44:17,839-<run>-[line:101]-INFO: patience_scheduler=5
# 2021-10-20 10:44:17,839-<run>-[line:102]-INFO: patience_early_stop=10
# 2021-10-20 10:44:17,839-<run>-[line:103]-INFO: ReduceLR_factor=0.2
# 2021-10-20 10:44:17,839-<run>-[line:105]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# 2021-10-20 10:44:17,994-<run>-[line:132]-INFO: ----------------------------------------- load train (33453): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# Parameters: 4104288
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [16, 24, 112, 112]             216
#        BatchNorm2d-2         [16, 24, 112, 112]              48
#               ReLU-3         [16, 24, 112, 112]               0
#             Conv2d-4         [16, 48, 112, 112]          10,368
#        BatchNorm2d-5         [16, 48, 112, 112]              96
#               ReLU-6         [16, 48, 112, 112]               0
#             Conv2d-7         [16, 10, 112, 112]           4,320
#        BatchNorm2d-8         [16, 10, 112, 112]              20
#               ReLU-9         [16, 10, 112, 112]               0
#            Conv2d-10         [16, 18, 112, 112]           9,396
#       BatchNorm2d-11         [16, 18, 112, 112]              36
#              ReLU-12         [16, 18, 112, 112]               0
#            Conv2d-13         [16, 10, 112, 112]           1,620
#       BatchNorm2d-14         [16, 10, 112, 112]              20
#              ReLU-15         [16, 10, 112, 112]               0
#            Conv2d-16         [16, 28, 112, 112]          19,152
#       BatchNorm2d-17         [16, 28, 112, 112]              56
#              ReLU-18         [16, 28, 112, 112]               0
#         HarDBlock-19         [16, 48, 112, 112]               0
#            Conv2d-20         [16, 64, 112, 112]           3,072
#       BatchNorm2d-21         [16, 64, 112, 112]             128
#              ReLU-22         [16, 64, 112, 112]               0
#         AvgPool2d-23           [16, 64, 56, 56]               0
#            Conv2d-24           [16, 16, 56, 56]           9,216
#       BatchNorm2d-25           [16, 16, 56, 56]              32
#              ReLU-26           [16, 16, 56, 56]               0
#            Conv2d-27           [16, 28, 56, 56]          20,160
#       BatchNorm2d-28           [16, 28, 56, 56]              56
#              ReLU-29           [16, 28, 56, 56]               0
#            Conv2d-30           [16, 16, 56, 56]           4,032
#       BatchNorm2d-31           [16, 16, 56, 56]              32
#              ReLU-32           [16, 16, 56, 56]               0
#            Conv2d-33           [16, 46, 56, 56]          44,712
#       BatchNorm2d-34           [16, 46, 56, 56]              92
#              ReLU-35           [16, 46, 56, 56]               0
#         HarDBlock-36           [16, 78, 56, 56]               0
#            Conv2d-37           [16, 96, 56, 56]           7,488
#       BatchNorm2d-38           [16, 96, 56, 56]             192
#              ReLU-39           [16, 96, 56, 56]               0
#         AvgPool2d-40           [16, 96, 28, 28]               0
#            Conv2d-41           [16, 18, 28, 28]          15,552
#       BatchNorm2d-42           [16, 18, 28, 28]              36
#              ReLU-43           [16, 18, 28, 28]               0
#            Conv2d-44           [16, 30, 28, 28]          30,780
#       BatchNorm2d-45           [16, 30, 28, 28]              60
#              ReLU-46           [16, 30, 28, 28]               0
#            Conv2d-47           [16, 18, 28, 28]           4,860
#       BatchNorm2d-48           [16, 18, 28, 28]              36
#              ReLU-49           [16, 18, 28, 28]               0
#            Conv2d-50           [16, 52, 28, 28]          67,392
#       BatchNorm2d-51           [16, 52, 28, 28]             104
#              ReLU-52           [16, 52, 28, 28]               0
#            Conv2d-53           [16, 18, 28, 28]           8,424
#       BatchNorm2d-54           [16, 18, 28, 28]              36
#              ReLU-55           [16, 18, 28, 28]               0
#            Conv2d-56           [16, 30, 28, 28]          18,900
#       BatchNorm2d-57           [16, 30, 28, 28]              60
#              ReLU-58           [16, 30, 28, 28]               0
#            Conv2d-59           [16, 18, 28, 28]           4,860
#       BatchNorm2d-60           [16, 18, 28, 28]              36
#              ReLU-61           [16, 18, 28, 28]               0
#            Conv2d-62           [16, 88, 28, 28]         155,232
#       BatchNorm2d-63           [16, 88, 28, 28]             176
#              ReLU-64           [16, 88, 28, 28]               0
#         HarDBlock-65          [16, 160, 28, 28]               0
#            Conv2d-66          [16, 160, 28, 28]          25,600
#       BatchNorm2d-67          [16, 160, 28, 28]             320
#              ReLU-68          [16, 160, 28, 28]               0
#         AvgPool2d-69          [16, 160, 14, 14]               0
#            Conv2d-70           [16, 24, 14, 14]          34,560
#       BatchNorm2d-71           [16, 24, 14, 14]              48
#              ReLU-72           [16, 24, 14, 14]               0
#            Conv2d-73           [16, 40, 14, 14]          66,240
#       BatchNorm2d-74           [16, 40, 14, 14]              80
#              ReLU-75           [16, 40, 14, 14]               0
#            Conv2d-76           [16, 24, 14, 14]           8,640
#       BatchNorm2d-77           [16, 24, 14, 14]              48
#              ReLU-78           [16, 24, 14, 14]               0
#            Conv2d-79           [16, 70, 14, 14]         141,120
#       BatchNorm2d-80           [16, 70, 14, 14]             140
#              ReLU-81           [16, 70, 14, 14]               0
#            Conv2d-82           [16, 24, 14, 14]          15,120
#       BatchNorm2d-83           [16, 24, 14, 14]              48
#              ReLU-84           [16, 24, 14, 14]               0
#            Conv2d-85           [16, 40, 14, 14]          33,840
#       BatchNorm2d-86           [16, 40, 14, 14]              80
#              ReLU-87           [16, 40, 14, 14]               0
#            Conv2d-88           [16, 24, 14, 14]           8,640
#       BatchNorm2d-89           [16, 24, 14, 14]              48
#              ReLU-90           [16, 24, 14, 14]               0
#            Conv2d-91          [16, 118, 14, 14]         312,228
#       BatchNorm2d-92          [16, 118, 14, 14]             236
#              ReLU-93          [16, 118, 14, 14]               0
#         HarDBlock-94          [16, 214, 14, 14]               0
#            Conv2d-95          [16, 224, 14, 14]          47,936
#       BatchNorm2d-96          [16, 224, 14, 14]             448
#              ReLU-97          [16, 224, 14, 14]               0
#         AvgPool2d-98            [16, 224, 7, 7]               0
#            Conv2d-99             [16, 32, 7, 7]          64,512
#      BatchNorm2d-100             [16, 32, 7, 7]              64
#             ReLU-101             [16, 32, 7, 7]               0
#           Conv2d-102             [16, 54, 7, 7]         124,416
#      BatchNorm2d-103             [16, 54, 7, 7]             108
#             ReLU-104             [16, 54, 7, 7]               0
#           Conv2d-105             [16, 32, 7, 7]          15,552
#      BatchNorm2d-106             [16, 32, 7, 7]              64
#             ReLU-107             [16, 32, 7, 7]               0
#           Conv2d-108             [16, 92, 7, 7]         256,680
#      BatchNorm2d-109             [16, 92, 7, 7]             184
#             ReLU-110             [16, 92, 7, 7]               0
#           Conv2d-111             [16, 32, 7, 7]          26,496
#      BatchNorm2d-112             [16, 32, 7, 7]              64
#             ReLU-113             [16, 32, 7, 7]               0
#           Conv2d-114             [16, 54, 7, 7]          60,264
#      BatchNorm2d-115             [16, 54, 7, 7]             108
#             ReLU-116             [16, 54, 7, 7]               0
#           Conv2d-117             [16, 32, 7, 7]          15,552
#      BatchNorm2d-118             [16, 32, 7, 7]              64
#             ReLU-119             [16, 32, 7, 7]               0
#           Conv2d-120            [16, 158, 7, 7]         571,644
#      BatchNorm2d-121            [16, 158, 7, 7]             316
#             ReLU-122            [16, 158, 7, 7]               0
#        HarDBlock-123            [16, 286, 7, 7]               0
#           Conv2d-124            [16, 320, 7, 7]          91,520
#      BatchNorm2d-125            [16, 320, 7, 7]             640
#             ReLU-126            [16, 320, 7, 7]               0
#     TransitionUp-127          [16, 534, 14, 14]               0
#           Conv2d-128          [16, 267, 14, 14]         142,578
#      BatchNorm2d-129          [16, 267, 14, 14]             534
#             ReLU-130          [16, 267, 14, 14]               0
#           Conv2d-131           [16, 24, 14, 14]          57,672
#      BatchNorm2d-132           [16, 24, 14, 14]              48
#             ReLU-133           [16, 24, 14, 14]               0
#           Conv2d-134           [16, 40, 14, 14]         104,760
#      BatchNorm2d-135           [16, 40, 14, 14]              80
#             ReLU-136           [16, 40, 14, 14]               0
#           Conv2d-137           [16, 24, 14, 14]           8,640
#      BatchNorm2d-138           [16, 24, 14, 14]              48
#             ReLU-139           [16, 24, 14, 14]               0
#           Conv2d-140           [16, 70, 14, 14]         208,530
#      BatchNorm2d-141           [16, 70, 14, 14]             140
#             ReLU-142           [16, 70, 14, 14]               0
#           Conv2d-143           [16, 24, 14, 14]          15,120
#      BatchNorm2d-144           [16, 24, 14, 14]              48
#             ReLU-145           [16, 24, 14, 14]               0
#           Conv2d-146           [16, 40, 14, 14]          33,840
#      BatchNorm2d-147           [16, 40, 14, 14]              80
#             ReLU-148           [16, 40, 14, 14]               0
#           Conv2d-149           [16, 24, 14, 14]           8,640
#      BatchNorm2d-150           [16, 24, 14, 14]              48
#             ReLU-151           [16, 24, 14, 14]               0
#           Conv2d-152          [16, 118, 14, 14]         425,862
#      BatchNorm2d-153          [16, 118, 14, 14]             236
#             ReLU-154          [16, 118, 14, 14]               0
#        HarDBlock-155          [16, 214, 14, 14]               0
#     TransitionUp-156          [16, 374, 28, 28]               0
#           Conv2d-157          [16, 187, 28, 28]          69,938
#      BatchNorm2d-158          [16, 187, 28, 28]             374
#             ReLU-159          [16, 187, 28, 28]               0
#           Conv2d-160           [16, 18, 28, 28]          30,294
#      BatchNorm2d-161           [16, 18, 28, 28]              36
#             ReLU-162           [16, 18, 28, 28]               0
#           Conv2d-163           [16, 30, 28, 28]          55,350
#      BatchNorm2d-164           [16, 30, 28, 28]              60
#             ReLU-165           [16, 30, 28, 28]               0
#           Conv2d-166           [16, 18, 28, 28]           4,860
#      BatchNorm2d-167           [16, 18, 28, 28]              36
#             ReLU-168           [16, 18, 28, 28]               0
#           Conv2d-169           [16, 52, 28, 28]         109,980
#      BatchNorm2d-170           [16, 52, 28, 28]             104
#             ReLU-171           [16, 52, 28, 28]               0
#           Conv2d-172           [16, 18, 28, 28]           8,424
#      BatchNorm2d-173           [16, 18, 28, 28]              36
#             ReLU-174           [16, 18, 28, 28]               0
#           Conv2d-175           [16, 30, 28, 28]          18,900
#      BatchNorm2d-176           [16, 30, 28, 28]              60
#             ReLU-177           [16, 30, 28, 28]               0
#           Conv2d-178           [16, 18, 28, 28]           4,860
#      BatchNorm2d-179           [16, 18, 28, 28]              36
#             ReLU-180           [16, 18, 28, 28]               0
#           Conv2d-181           [16, 88, 28, 28]         227,304
#      BatchNorm2d-182           [16, 88, 28, 28]             176
#             ReLU-183           [16, 88, 28, 28]               0
#        HarDBlock-184          [16, 160, 28, 28]               0
#     TransitionUp-185          [16, 238, 56, 56]               0
#           Conv2d-186          [16, 119, 56, 56]          28,322
#      BatchNorm2d-187          [16, 119, 56, 56]             238
#             ReLU-188          [16, 119, 56, 56]               0
#           Conv2d-189           [16, 16, 56, 56]          17,136
#      BatchNorm2d-190           [16, 16, 56, 56]              32
#             ReLU-191           [16, 16, 56, 56]               0
#           Conv2d-192           [16, 28, 56, 56]          34,020
#      BatchNorm2d-193           [16, 28, 56, 56]              56
#             ReLU-194           [16, 28, 56, 56]               0
#           Conv2d-195           [16, 16, 56, 56]           4,032
#      BatchNorm2d-196           [16, 16, 56, 56]              32
#             ReLU-197           [16, 16, 56, 56]               0
#           Conv2d-198           [16, 46, 56, 56]          67,482
#      BatchNorm2d-199           [16, 46, 56, 56]              92
#             ReLU-200           [16, 46, 56, 56]               0
#        HarDBlock-201           [16, 78, 56, 56]               0
#     TransitionUp-202        [16, 126, 112, 112]               0
#           Conv2d-203         [16, 63, 112, 112]           7,938
#      BatchNorm2d-204         [16, 63, 112, 112]             126
#             ReLU-205         [16, 63, 112, 112]               0
#           Conv2d-206         [16, 10, 112, 112]           5,670
#      BatchNorm2d-207         [16, 10, 112, 112]              20
#             ReLU-208         [16, 10, 112, 112]               0
#           Conv2d-209         [16, 18, 112, 112]          11,826
#      BatchNorm2d-210         [16, 18, 112, 112]              36
#             ReLU-211         [16, 18, 112, 112]               0
#           Conv2d-212         [16, 10, 112, 112]           1,620
#      BatchNorm2d-213         [16, 10, 112, 112]              20
#             ReLU-214         [16, 10, 112, 112]               0
#           Conv2d-215         [16, 28, 112, 112]          22,932
#      BatchNorm2d-216         [16, 28, 112, 112]              56
#             ReLU-217         [16, 28, 112, 112]               0
#        HarDBlock-218         [16, 48, 112, 112]               0
#           Conv2d-219          [16, 2, 112, 112]              98
# ================================================================
# Total params: 4,104,288
# Trainable params: 4,104,288
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.06
# Forward/backward pass size (MB): 2992.52
# Params size (MB): 15.66
# Estimated Total Size (MB): 3011.24
# ----------------------------------------------------------------
# Parameters: 4104288
# predict 100 imgs : 4.139642715454102
# predict 100 imgs : 0:00:04.139642
# predict FPS is      : 24.15667410781136/s
# ------------------------------------------------
# predict 1,000 imgs : 6.470919370651245
# predict 1,000 imgs : 0:00:06.470920
# predict FPS is      : 154.53754601478803/s
# ------------------------------------------------
# predict 10,000 imgs : 32.143367290496826
# predict 10,000 imgs : 0:00:32.143367
# predict FPS is      : 311.1061734641752/s
# ------------------------------------------------
# predict 30,000 imgs : 89.28609085083008
# predict 30,000 imgs : 0:01:29.286091
# predict FPS is      : 335.9985829161328/s
# ------------------------------------------------
# all:134.80835914611816
# all:0:02:14.808359
#
# Process finished with exit code 0














