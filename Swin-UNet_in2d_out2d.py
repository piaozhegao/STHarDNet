

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
from compare_models_speed.ATLAS_models.lib.c_hardnet_swin_ende_decoder import SwinTransformerSys


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
    logging.info("------------------------{}".format(logdir))
    logger = get_logger(logdir)
    logger.info("Let the games begin")

    ## save model log and check points
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


    # train_data = DataLoader(CustomDataset2_gray(train_data_list[:img_count], train_mask_list[:img_count]), batch_size=batch_size, shuffle=True, num_workers=4)
    # val_data   = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = DiceCE()
    criterion.to(device)

    ## create model
    ## create model
    embed_dim = 96  # 4*4*1*2=32  ## 4*4*3*2=96
    depths = [2, 2, 2, 2]
    depths_decoder = [1, 2, 2, 2]
    num_heads = [3, 6, 12, 24]
    mlp_ratio = 4.
    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.
    drop_path_rate = 0.1
    ape = False
    patch_norm = True
    use_checkpoint = False
    img_size = 224
    patch_size = 4
    in_chans = 1
    num_classes = 2
    window_size = 7
    model = SwinTransformerSys(img_size=img_size,
                               patch_size=patch_size,
                               in_chans=in_chans,
                               num_classes=num_classes,
                               embed_dim=embed_dim,
                               depths=depths,
                               num_heads=num_heads,
                               window_size=window_size,
                               mlp_ratio=mlp_ratio,
                               qkv_bias=qkv_bias,
                               qk_scale=qk_scale,
                               drop_rate=drop_rate,
                               drop_path_rate=drop_path_rate,
                               ape=ape,
                               patch_norm=patch_norm,
                               use_checkpoint=use_checkpoint).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Parameters:'.format(total_params))


    ##  load pre-trained model
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)

    summary_(model, (1, 224, 224), batch_size=batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info('Parameters:'.format(total_params))

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

    #########################################
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
    model_path = "C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/models/swin_unet_in2d_out2d"
    model_path = os.path.join(model_path, "c_swin_unet_ATLAS_best_model.pkl")
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
# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/Swin-UNet_in2d_out2d.py
# 2021-10-20 10:24:31,614-<run>-[line:91]-INFO: Let the games begin
# 2021-10-20 10:24:31,614-<run>-[line:92]-INFO: ========================================
# 2021-10-20 10:24:31,614-<run>-[line:93]-INFO: # parameters:
# 2021-10-20 10:24:31,614-<run>-[line:94]-INFO: IMG_H=224
# 2021-10-20 10:24:31,614-<run>-[line:95]-INFO: IMG_W=224
# 2021-10-20 10:24:31,614-<run>-[line:96]-INFO: img_depth=1
# 2021-10-20 10:24:31,614-<run>-[line:97]-INFO: img_depth_out_num=2
# 2021-10-20 10:24:31,614-<run>-[line:98]-INFO: batch_size=16
# 2021-10-20 10:24:31,615-<run>-[line:99]-INFO: n_classes=2
# 2021-10-20 10:24:31,615-<run>-[line:100]-INFO: learning_rate=0.001
# 2021-10-20 10:24:31,615-<run>-[line:101]-INFO: patience_scheduler=5
# 2021-10-20 10:24:31,615-<run>-[line:102]-INFO: patience_early_stop=10
# 2021-10-20 10:24:31,615-<run>-[line:103]-INFO: ReduceLR_factor=0.2
# 2021-10-20 10:24:31,615-<run>-[line:105]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# 2021-10-20 10:24:31,778-<run>-[line:132]-INFO: ----------------------------------------- load train (33453): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# SwinTransformerSys expand initial----depths:[2, 2, 2, 2];depths_decoder:[1, 2, 2, 2];drop_path_rate:0.1;num_classes:2
# ---final upsample expand_first---
# Parameters: 27165156
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [16, 96, 56, 56]           1,632
#          LayerNorm-2             [16, 3136, 96]             192
#         PatchEmbed-3             [16, 3136, 96]               0
#            Dropout-4             [16, 3136, 96]               0
#          LayerNorm-5             [16, 3136, 96]             192
#             Linear-6              [16, 49, 288]          27,936
#            Softmax-7            [16, 3, 49, 49]               0
#            Dropout-8            [16, 3, 49, 49]               0
#             Linear-9               [16, 49, 96]           9,312
#           Dropout-10               [16, 49, 96]               0
#   WindowAttention-11               [16, 49, 96]               0
#          Identity-12             [16, 3136, 96]               0
#         LayerNorm-13             [16, 3136, 96]             192
#            Linear-14            [16, 3136, 384]          37,248
#              GELU-15            [16, 3136, 384]               0
#           Dropout-16            [16, 3136, 384]               0
#            Linear-17             [16, 3136, 96]          36,960
#           Dropout-18             [16, 3136, 96]               0
#               Mlp-19             [16, 3136, 96]               0
#          Identity-20             [16, 3136, 96]               0
# SwinTransformerBlock-21             [16, 3136, 96]               0
#         LayerNorm-22             [16, 3136, 96]             192
#            Linear-23              [16, 49, 288]          27,936
#           Softmax-24            [16, 3, 49, 49]               0
#           Dropout-25            [16, 3, 49, 49]               0
#            Linear-26               [16, 49, 96]           9,312
#           Dropout-27               [16, 49, 96]               0
#   WindowAttention-28               [16, 49, 96]               0
#          DropPath-29             [16, 3136, 96]               0
#         LayerNorm-30             [16, 3136, 96]             192
#            Linear-31            [16, 3136, 384]          37,248
#              GELU-32            [16, 3136, 384]               0
#           Dropout-33            [16, 3136, 384]               0
#            Linear-34             [16, 3136, 96]          36,960
#           Dropout-35             [16, 3136, 96]               0
#               Mlp-36             [16, 3136, 96]               0
#          DropPath-37             [16, 3136, 96]               0
# SwinTransformerBlock-38             [16, 3136, 96]               0
#         LayerNorm-39             [16, 784, 384]             768
#            Linear-40             [16, 784, 192]          73,728
#      PatchMerging-41             [16, 784, 192]               0
#        BasicLayer-42             [16, 784, 192]               0
#         LayerNorm-43             [16, 784, 192]             384
#            Linear-44              [16, 49, 576]         111,168
#           Softmax-45            [16, 6, 49, 49]               0
#           Dropout-46            [16, 6, 49, 49]               0
#            Linear-47              [16, 49, 192]          37,056
#           Dropout-48              [16, 49, 192]               0
#   WindowAttention-49              [16, 49, 192]               0
#          DropPath-50             [16, 784, 192]               0
#         LayerNorm-51             [16, 784, 192]             384
#            Linear-52             [16, 784, 768]         148,224
#              GELU-53             [16, 784, 768]               0
#           Dropout-54             [16, 784, 768]               0
#            Linear-55             [16, 784, 192]         147,648
#           Dropout-56             [16, 784, 192]               0
#               Mlp-57             [16, 784, 192]               0
#          DropPath-58             [16, 784, 192]               0
# SwinTransformerBlock-59             [16, 784, 192]               0
#         LayerNorm-60             [16, 784, 192]             384
#            Linear-61              [16, 49, 576]         111,168
#           Softmax-62            [16, 6, 49, 49]               0
#           Dropout-63            [16, 6, 49, 49]               0
#            Linear-64              [16, 49, 192]          37,056
#           Dropout-65              [16, 49, 192]               0
#   WindowAttention-66              [16, 49, 192]               0
#          DropPath-67             [16, 784, 192]               0
#         LayerNorm-68             [16, 784, 192]             384
#            Linear-69             [16, 784, 768]         148,224
#              GELU-70             [16, 784, 768]               0
#           Dropout-71             [16, 784, 768]               0
#            Linear-72             [16, 784, 192]         147,648
#           Dropout-73             [16, 784, 192]               0
#               Mlp-74             [16, 784, 192]               0
#          DropPath-75             [16, 784, 192]               0
# SwinTransformerBlock-76             [16, 784, 192]               0
#         LayerNorm-77             [16, 196, 768]           1,536
#            Linear-78             [16, 196, 384]         294,912
#      PatchMerging-79             [16, 196, 384]               0
#        BasicLayer-80             [16, 196, 384]               0
#         LayerNorm-81             [16, 196, 384]             768
#            Linear-82             [16, 49, 1152]         443,520
#           Softmax-83           [16, 12, 49, 49]               0
#           Dropout-84           [16, 12, 49, 49]               0
#            Linear-85              [16, 49, 384]         147,840
#           Dropout-86              [16, 49, 384]               0
#   WindowAttention-87              [16, 49, 384]               0
#          DropPath-88             [16, 196, 384]               0
#         LayerNorm-89             [16, 196, 384]             768
#            Linear-90            [16, 196, 1536]         591,360
#              GELU-91            [16, 196, 1536]               0
#           Dropout-92            [16, 196, 1536]               0
#            Linear-93             [16, 196, 384]         590,208
#           Dropout-94             [16, 196, 384]               0
#               Mlp-95             [16, 196, 384]               0
#          DropPath-96             [16, 196, 384]               0
# SwinTransformerBlock-97             [16, 196, 384]               0
#         LayerNorm-98             [16, 196, 384]             768
#            Linear-99             [16, 49, 1152]         443,520
#          Softmax-100           [16, 12, 49, 49]               0
#          Dropout-101           [16, 12, 49, 49]               0
#           Linear-102              [16, 49, 384]         147,840
#          Dropout-103              [16, 49, 384]               0
#  WindowAttention-104              [16, 49, 384]               0
#         DropPath-105             [16, 196, 384]               0
#        LayerNorm-106             [16, 196, 384]             768
#           Linear-107            [16, 196, 1536]         591,360
#             GELU-108            [16, 196, 1536]               0
#          Dropout-109            [16, 196, 1536]               0
#           Linear-110             [16, 196, 384]         590,208
#          Dropout-111             [16, 196, 384]               0
#              Mlp-112             [16, 196, 384]               0
#         DropPath-113             [16, 196, 384]               0
# SwinTransformerBlock-114             [16, 196, 384]               0
#        LayerNorm-115             [16, 49, 1536]           3,072
#           Linear-116              [16, 49, 768]       1,179,648
#     PatchMerging-117              [16, 49, 768]               0
#       BasicLayer-118              [16, 49, 768]               0
#        LayerNorm-119              [16, 49, 768]           1,536
#           Linear-120             [16, 49, 2304]       1,771,776
#          Softmax-121           [16, 24, 49, 49]               0
#          Dropout-122           [16, 24, 49, 49]               0
#           Linear-123              [16, 49, 768]         590,592
#          Dropout-124              [16, 49, 768]               0
#  WindowAttention-125              [16, 49, 768]               0
#         DropPath-126              [16, 49, 768]               0
#        LayerNorm-127              [16, 49, 768]           1,536
#           Linear-128             [16, 49, 3072]       2,362,368
#             GELU-129             [16, 49, 3072]               0
#          Dropout-130             [16, 49, 3072]               0
#           Linear-131              [16, 49, 768]       2,360,064
#          Dropout-132              [16, 49, 768]               0
#              Mlp-133              [16, 49, 768]               0
#         DropPath-134              [16, 49, 768]               0
# SwinTransformerBlock-135              [16, 49, 768]               0
#        LayerNorm-136              [16, 49, 768]           1,536
#           Linear-137             [16, 49, 2304]       1,771,776
#          Softmax-138           [16, 24, 49, 49]               0
#          Dropout-139           [16, 24, 49, 49]               0
#           Linear-140              [16, 49, 768]         590,592
#          Dropout-141              [16, 49, 768]               0
#  WindowAttention-142              [16, 49, 768]               0
#         DropPath-143              [16, 49, 768]               0
#        LayerNorm-144              [16, 49, 768]           1,536
#           Linear-145             [16, 49, 3072]       2,362,368
#             GELU-146             [16, 49, 3072]               0
#          Dropout-147             [16, 49, 3072]               0
#           Linear-148              [16, 49, 768]       2,360,064
#          Dropout-149              [16, 49, 768]               0
#              Mlp-150              [16, 49, 768]               0
#         DropPath-151              [16, 49, 768]               0
# SwinTransformerBlock-152              [16, 49, 768]               0
#       BasicLayer-153              [16, 49, 768]               0
#        LayerNorm-154              [16, 49, 768]           1,536
#           Linear-155             [16, 49, 1536]       1,179,648
#        LayerNorm-156             [16, 196, 384]             768
#      PatchExpand-157             [16, 196, 384]               0
#           Linear-158             [16, 196, 384]         295,296
#        LayerNorm-159             [16, 196, 384]             768
#           Linear-160             [16, 49, 1152]         443,520
#          Softmax-161           [16, 12, 49, 49]               0
#          Dropout-162           [16, 12, 49, 49]               0
#           Linear-163              [16, 49, 384]         147,840
#          Dropout-164              [16, 49, 384]               0
#  WindowAttention-165              [16, 49, 384]               0
#         DropPath-166             [16, 196, 384]               0
#        LayerNorm-167             [16, 196, 384]             768
#           Linear-168            [16, 196, 1536]         591,360
#             GELU-169            [16, 196, 1536]               0
#          Dropout-170            [16, 196, 1536]               0
#           Linear-171             [16, 196, 384]         590,208
#          Dropout-172             [16, 196, 384]               0
#              Mlp-173             [16, 196, 384]               0
#         DropPath-174             [16, 196, 384]               0
# SwinTransformerBlock-175             [16, 196, 384]               0
#        LayerNorm-176             [16, 196, 384]             768
#           Linear-177             [16, 49, 1152]         443,520
#          Softmax-178           [16, 12, 49, 49]               0
#          Dropout-179           [16, 12, 49, 49]               0
#           Linear-180              [16, 49, 384]         147,840
#          Dropout-181              [16, 49, 384]               0
#  WindowAttention-182              [16, 49, 384]               0
#         DropPath-183             [16, 196, 384]               0
#        LayerNorm-184             [16, 196, 384]             768
#           Linear-185            [16, 196, 1536]         591,360
#             GELU-186            [16, 196, 1536]               0
#          Dropout-187            [16, 196, 1536]               0
#           Linear-188             [16, 196, 384]         590,208
#          Dropout-189             [16, 196, 384]               0
#              Mlp-190             [16, 196, 384]               0
#         DropPath-191             [16, 196, 384]               0
# SwinTransformerBlock-192             [16, 196, 384]               0
#           Linear-193             [16, 196, 768]         294,912
#        LayerNorm-194             [16, 784, 192]             384
#      PatchExpand-195             [16, 784, 192]               0
#    BasicLayer_up-196             [16, 784, 192]               0
#           Linear-197             [16, 784, 192]          73,920
#        LayerNorm-198             [16, 784, 192]             384
#           Linear-199              [16, 49, 576]         111,168
#          Softmax-200            [16, 6, 49, 49]               0
#          Dropout-201            [16, 6, 49, 49]               0
#           Linear-202              [16, 49, 192]          37,056
#          Dropout-203              [16, 49, 192]               0
#  WindowAttention-204              [16, 49, 192]               0
#         DropPath-205             [16, 784, 192]               0
#        LayerNorm-206             [16, 784, 192]             384
#           Linear-207             [16, 784, 768]         148,224
#             GELU-208             [16, 784, 768]               0
#          Dropout-209             [16, 784, 768]               0
#           Linear-210             [16, 784, 192]         147,648
#          Dropout-211             [16, 784, 192]               0
#              Mlp-212             [16, 784, 192]               0
#         DropPath-213             [16, 784, 192]               0
# SwinTransformerBlock-214             [16, 784, 192]               0
#        LayerNorm-215             [16, 784, 192]             384
#           Linear-216              [16, 49, 576]         111,168
#          Softmax-217            [16, 6, 49, 49]               0
#          Dropout-218            [16, 6, 49, 49]               0
#           Linear-219              [16, 49, 192]          37,056
#          Dropout-220              [16, 49, 192]               0
#  WindowAttention-221              [16, 49, 192]               0
#         DropPath-222             [16, 784, 192]               0
#        LayerNorm-223             [16, 784, 192]             384
#           Linear-224             [16, 784, 768]         148,224
#             GELU-225             [16, 784, 768]               0
#          Dropout-226             [16, 784, 768]               0
#           Linear-227             [16, 784, 192]         147,648
#          Dropout-228             [16, 784, 192]               0
#              Mlp-229             [16, 784, 192]               0
#         DropPath-230             [16, 784, 192]               0
# SwinTransformerBlock-231             [16, 784, 192]               0
#           Linear-232             [16, 784, 384]          73,728
#        LayerNorm-233             [16, 3136, 96]             192
#      PatchExpand-234             [16, 3136, 96]               0
#    BasicLayer_up-235             [16, 3136, 96]               0
#           Linear-236             [16, 3136, 96]          18,528
#        LayerNorm-237             [16, 3136, 96]             192
#           Linear-238              [16, 49, 288]          27,936
#          Softmax-239            [16, 3, 49, 49]               0
#          Dropout-240            [16, 3, 49, 49]               0
#           Linear-241               [16, 49, 96]           9,312
#          Dropout-242               [16, 49, 96]               0
#  WindowAttention-243               [16, 49, 96]               0
#         Identity-244             [16, 3136, 96]               0
#        LayerNorm-245             [16, 3136, 96]             192
#           Linear-246            [16, 3136, 384]          37,248
#             GELU-247            [16, 3136, 384]               0
#          Dropout-248            [16, 3136, 384]               0
#           Linear-249             [16, 3136, 96]          36,960
#          Dropout-250             [16, 3136, 96]               0
#              Mlp-251             [16, 3136, 96]               0
#         Identity-252             [16, 3136, 96]               0
# SwinTransformerBlock-253             [16, 3136, 96]               0
#        LayerNorm-254             [16, 3136, 96]             192
#           Linear-255              [16, 49, 288]          27,936
#          Softmax-256            [16, 3, 49, 49]               0
#          Dropout-257            [16, 3, 49, 49]               0
#           Linear-258               [16, 49, 96]           9,312
#          Dropout-259               [16, 49, 96]               0
#  WindowAttention-260               [16, 49, 96]               0
#         DropPath-261             [16, 3136, 96]               0
#        LayerNorm-262             [16, 3136, 96]             192
#           Linear-263            [16, 3136, 384]          37,248
#             GELU-264            [16, 3136, 384]               0
#          Dropout-265            [16, 3136, 384]               0
#           Linear-266             [16, 3136, 96]          36,960
#          Dropout-267             [16, 3136, 96]               0
#              Mlp-268             [16, 3136, 96]               0
#         DropPath-269             [16, 3136, 96]               0
# SwinTransformerBlock-270             [16, 3136, 96]               0
#    BasicLayer_up-271             [16, 3136, 96]               0
#        LayerNorm-272             [16, 3136, 96]             192
#           Linear-273           [16, 3136, 1536]         147,456
#        LayerNorm-274            [16, 50176, 96]             192
# FinalPatchExpand_X4-275            [16, 50176, 96]               0
#           Conv2d-276          [16, 2, 224, 224]             192
# ================================================================
# Total params: 27,142,848
# Trainable params: 27,142,848
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 3.06
# Forward/backward pass size (MB): 8036.81
# Params size (MB): 103.54
# Estimated Total Size (MB): 8143.42
# ----------------------------------------------------------------
# Parameters: 27165156
# predict 100 imgs : 3.736844301223755
# predict 100 imgs : 0:00:03.736845
# predict FPS is      : 26.760547654407663/s
# ------------------------------------------------
# predict 1,000 imgs : 6.962219715118408
# predict 1,000 imgs : 0:00:06.962220
# predict FPS is      : 143.6323530308168/s
# ------------------------------------------------
# predict 10,000 imgs : 41.3510057926178
# predict 10,000 imgs : 0:00:41.351005
# predict FPS is      : 241.8320862653661/s
# ------------------------------------------------
# predict 30,000 imgs : 122.50359988212585
# predict 30,000 imgs : 0:02:02.503600
# predict FPS is      : 244.89076262955774/s
# ------------------------------------------------
# all:177.38307547569275
# all:0:02:57.383076
#
# Process finished with exit code 0









