

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

from compare_models_speed.ATLAS_models.lib.loader import CustomDataset2
from c_ATLAS.c_2d_3d_swin_hardnet.logger import get_logger
from compare_models_speed.ATLAS_models.lib.loss import DiceCE
# from c_ATLAS.c_2d_3d_swin_hardnet.hardnet_2d_3d import swin_hardnet_2d_3d
from compare_models_speed.ATLAS_models.lib.c_metrics import get_score_from_all_slices_cherhoo_depth
from compare_models_speed.ATLAS_models.lib.swin_hardnet import SwinTransformerSegmentation


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
    train_data_list = [train_img_path + i for i in os.listdir(train_img_path)]
    train_mask_list = [train_mask_path + i for i in os.listdir(train_mask_path)]
    # val_data_list = [val_img_path + i for i in os.listdir(val_img_path)]
    # val_mask_list = [val_mask_path + i for i in os.listdir(val_mask_path)]
    train_data_list.extend(train_data_list)
    train_data_list.extend(train_data_list)
    train_mask_list.extend(train_mask_list)
    train_mask_list.extend(train_mask_list)
    logger.info("----------------------------------------- load train ({}): {}".format(len(train_data_list), train_img_path))


    train_data = DataLoader(CustomDataset2(train_data_list[:img_count], train_mask_list[:img_count]), batch_size=batch_size, shuffle=True, num_workers=4)
    # val_data   = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W, out_num=img_depth_out_num), batch_size=batch_size, shuffle=True, num_workers=4)

    criterion = DiceCE()
    criterion.to(device)

    ## create model
    model = SwinTransformerSegmentation(img_size=224,
                                        patch_size=4,
                                        in_chans=3,
                                        num_classes=2,
                                        embed_dim=96,
                                        # embed_dim=48,
                                        depths=[2, 2, 6, 2],
                                        num_heads=[3, 6, 12, 24],
                                        window_size=7,
                                        mlp_ratio=4.,
                                        qkv_bias=True,
                                        qk_scale=None,
                                        drop_rate=0.,
                                        drop_path_rate=0.1,
                                        ape=False,
                                        patch_norm=True,
                                        use_checkpoint=False).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)


    ##  load pre-trained model
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)
    #summary_(model, (3, 224, 224), batch_size=batch_size)
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_data = DataLoader(CustomDataset2(train_data_list[:100], train_mask_list[:100]), batch_size=batch_size, shuffle=True, num_workers=4)
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

    train_data = DataLoader(CustomDataset2(train_data_list[:1000], train_mask_list[:1000]), batch_size=batch_size, shuffle=True, num_workers=4)
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

    train_data = DataLoader(CustomDataset2(train_data_list[:10000], train_mask_list[:10000]), batch_size=batch_size, shuffle=True, num_workers=4)
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

    train_data = DataLoader(CustomDataset2(train_data_list[:30000], train_mask_list[:30000]), batch_size=batch_size, shuffle=True, num_workers=4)
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


    #######################################################
    train_data = DataLoader(CustomDataset2(train_data_list[:50000], train_mask_list[:50000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    print("predict 50,000 imgs : {}".format(runing_time))
    print("predict 50,000 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(50000 / runing_time))
    print("------------------------------------------------")

    train_data = DataLoader(CustomDataset2(train_data_list[:100000], train_mask_list[:100000]), batch_size=batch_size, shuffle=True, num_workers=4)
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
    print("predict 100,000 imgs : {}".format(runing_time))
    print("predict 100,000 imgs : {}".format(runing_time_2))
    print("predict FPS is      : {}/s".format(100000 / runing_time))
    print("------------------------------------------------")


    return 0



import time
import datetime
if __name__ == '__main__':
    torch.cuda.empty_cache()
    test_score_list = []
    start = time.time()
    start_2 = datetime.datetime.now()
    model_path = r"C:\Users\user\Documents\python-code\11_c_paper\compare_models_speed\ATLAS_models\models\SwinTransformer_in2d_out2d"
    model_path = os.path.join(model_path, "c_swin_transformer_segmentation_224_best_model.pkl")
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
# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/SwinTransformer_in2d_out2d.py
# 2021-10-22 00:06:10,095-<run>-[line:93]-INFO: Let the games begin
# 2021-10-22 00:06:10,095-<run>-[line:94]-INFO: ========================================
# 2021-10-22 00:06:10,096-<run>-[line:95]-INFO: # parameters:
# 2021-10-22 00:06:10,096-<run>-[line:96]-INFO: IMG_H=224
# 2021-10-22 00:06:10,096-<run>-[line:97]-INFO: IMG_W=224
# 2021-10-22 00:06:10,096-<run>-[line:98]-INFO: img_depth=1
# 2021-10-22 00:06:10,096-<run>-[line:99]-INFO: img_depth_out_num=2
# 2021-10-22 00:06:10,096-<run>-[line:100]-INFO: batch_size=16
# 2021-10-22 00:06:10,096-<run>-[line:101]-INFO: n_classes=2
# 2021-10-22 00:06:10,096-<run>-[line:102]-INFO: learning_rate=0.001
# 2021-10-22 00:06:10,096-<run>-[line:103]-INFO: patience_scheduler=5
# 2021-10-22 00:06:10,096-<run>-[line:104]-INFO: patience_early_stop=10
# 2021-10-22 00:06:10,096-<run>-[line:105]-INFO: ReduceLR_factor=0.2
# 2021-10-22 00:06:10,096-<run>-[line:107]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# SwinTransformerSys expand initial----depths:[2, 2, 6, 2];depths_decoder:[1, 2, 2, 2];drop_path_rate:0.1;num_classes:2
# 2021-10-22 00:06:10,246-<run>-[line:134]-INFO: ----------------------------------------- load train (33453): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# Parameters: 29878848
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [16, 96, 56, 56]           4,704
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
#    BasicLayer_new-42             [16, 784, 192]               0
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
#    BasicLayer_new-80             [16, 196, 384]               0
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
#        LayerNorm-115             [16, 196, 384]             768
#           Linear-116             [16, 49, 1152]         443,520
#          Softmax-117           [16, 12, 49, 49]               0
#          Dropout-118           [16, 12, 49, 49]               0
#           Linear-119              [16, 49, 384]         147,840
#          Dropout-120              [16, 49, 384]               0
#  WindowAttention-121              [16, 49, 384]               0
#         DropPath-122             [16, 196, 384]               0
#        LayerNorm-123             [16, 196, 384]             768
#           Linear-124            [16, 196, 1536]         591,360
#             GELU-125            [16, 196, 1536]               0
#          Dropout-126            [16, 196, 1536]               0
#           Linear-127             [16, 196, 384]         590,208
#          Dropout-128             [16, 196, 384]               0
#              Mlp-129             [16, 196, 384]               0
#         DropPath-130             [16, 196, 384]               0
# SwinTransformerBlock-131             [16, 196, 384]               0
#        LayerNorm-132             [16, 196, 384]             768
#           Linear-133             [16, 49, 1152]         443,520
#          Softmax-134           [16, 12, 49, 49]               0
#          Dropout-135           [16, 12, 49, 49]               0
#           Linear-136              [16, 49, 384]         147,840
#          Dropout-137              [16, 49, 384]               0
#  WindowAttention-138              [16, 49, 384]               0
#         DropPath-139             [16, 196, 384]               0
#        LayerNorm-140             [16, 196, 384]             768
#           Linear-141            [16, 196, 1536]         591,360
#             GELU-142            [16, 196, 1536]               0
#          Dropout-143            [16, 196, 1536]               0
#           Linear-144             [16, 196, 384]         590,208
#          Dropout-145             [16, 196, 384]               0
#              Mlp-146             [16, 196, 384]               0
#         DropPath-147             [16, 196, 384]               0
# SwinTransformerBlock-148             [16, 196, 384]               0
#        LayerNorm-149             [16, 196, 384]             768
#           Linear-150             [16, 49, 1152]         443,520
#          Softmax-151           [16, 12, 49, 49]               0
#          Dropout-152           [16, 12, 49, 49]               0
#           Linear-153              [16, 49, 384]         147,840
#          Dropout-154              [16, 49, 384]               0
#  WindowAttention-155              [16, 49, 384]               0
#         DropPath-156             [16, 196, 384]               0
#        LayerNorm-157             [16, 196, 384]             768
#           Linear-158            [16, 196, 1536]         591,360
#             GELU-159            [16, 196, 1536]               0
#          Dropout-160            [16, 196, 1536]               0
#           Linear-161             [16, 196, 384]         590,208
#          Dropout-162             [16, 196, 384]               0
#              Mlp-163             [16, 196, 384]               0
#         DropPath-164             [16, 196, 384]               0
# SwinTransformerBlock-165             [16, 196, 384]               0
#        LayerNorm-166             [16, 196, 384]             768
#           Linear-167             [16, 49, 1152]         443,520
#          Softmax-168           [16, 12, 49, 49]               0
#          Dropout-169           [16, 12, 49, 49]               0
#           Linear-170              [16, 49, 384]         147,840
#          Dropout-171              [16, 49, 384]               0
#  WindowAttention-172              [16, 49, 384]               0
#         DropPath-173             [16, 196, 384]               0
#        LayerNorm-174             [16, 196, 384]             768
#           Linear-175            [16, 196, 1536]         591,360
#             GELU-176            [16, 196, 1536]               0
#          Dropout-177            [16, 196, 1536]               0
#           Linear-178             [16, 196, 384]         590,208
#          Dropout-179             [16, 196, 384]               0
#              Mlp-180             [16, 196, 384]               0
#         DropPath-181             [16, 196, 384]               0
# SwinTransformerBlock-182             [16, 196, 384]               0
#        LayerNorm-183             [16, 49, 1536]           3,072
#           Linear-184              [16, 49, 768]       1,179,648
#     PatchMerging-185              [16, 49, 768]               0
#   BasicLayer_new-186              [16, 49, 768]               0
#        LayerNorm-187              [16, 49, 768]           1,536
#           Linear-188             [16, 49, 2304]       1,771,776
#          Softmax-189           [16, 24, 49, 49]               0
#          Dropout-190           [16, 24, 49, 49]               0
#           Linear-191              [16, 49, 768]         590,592
#          Dropout-192              [16, 49, 768]               0
#  WindowAttention-193              [16, 49, 768]               0
#         DropPath-194              [16, 49, 768]               0
#        LayerNorm-195              [16, 49, 768]           1,536
#           Linear-196             [16, 49, 3072]       2,362,368
#             GELU-197             [16, 49, 3072]               0
#          Dropout-198             [16, 49, 3072]               0
#           Linear-199              [16, 49, 768]       2,360,064
#          Dropout-200              [16, 49, 768]               0
#              Mlp-201              [16, 49, 768]               0
#         DropPath-202              [16, 49, 768]               0
# SwinTransformerBlock-203              [16, 49, 768]               0
#        LayerNorm-204              [16, 49, 768]           1,536
#           Linear-205             [16, 49, 2304]       1,771,776
#          Softmax-206           [16, 24, 49, 49]               0
#          Dropout-207           [16, 24, 49, 49]               0
#           Linear-208              [16, 49, 768]         590,592
#          Dropout-209              [16, 49, 768]               0
#  WindowAttention-210              [16, 49, 768]               0
#         DropPath-211              [16, 49, 768]               0
#        LayerNorm-212              [16, 49, 768]           1,536
#           Linear-213             [16, 49, 3072]       2,362,368
#             GELU-214             [16, 49, 3072]               0
#          Dropout-215             [16, 49, 3072]               0
#           Linear-216              [16, 49, 768]       2,360,064
#          Dropout-217              [16, 49, 768]               0
#              Mlp-218              [16, 49, 768]               0
#         DropPath-219              [16, 49, 768]               0
# SwinTransformerBlock-220              [16, 49, 768]               0
#   BasicLayer_new-221              [16, 49, 768]               0
#        LayerNorm-222              [16, 49, 768]           1,536
#           Linear-223             [16, 49, 3072]       2,359,296
#           Conv2d-224          [16, 2, 224, 224]               6
# ================================================================
# Total params: 29,855,334
# Trainable params: 29,855,334
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 9.19
# Forward/backward pass size (MB): 4078.22
# Params size (MB): 113.89
# Estimated Total Size (MB): 4201.30
# ----------------------------------------------------------------
# Parameters: 29878848
# predict 100 imgs : 4.7032434940338135
# predict 100 imgs : 0:00:04.703243
# predict FPS is      : 21.261922783043786/s
# ------------------------------------------------
# predict 1,000 imgs : 16.286082983016968
# predict 1,000 imgs : 0:00:16.286083
# predict FPS is      : 61.402118670449745/s
# ------------------------------------------------
# predict 10,000 imgs : 133.0243775844574
# predict 10,000 imgs : 0:02:13.024378
# predict FPS is      : 75.1741912391282/s
# ------------------------------------------------
# predict 30,000 imgs : 390.87879943847656
# predict 30,000 imgs : 0:06:30.878799
# predict FPS is      : 76.75013339965483/s
# ------------------------------------------------
# all:547.6730990409851
# all:0:09:07.673099
#
# Process finished with exit code 0







# C:\Users\user\.conda\envs\open-mmlab\python.exe C:/Users/user/Documents/python-code/11_c_paper/compare_models_speed/ATLAS_models/SwinTransformer_in2d_out2d.py
# 2021-10-24 15:25:02,234-<run>-[line:93]-INFO: Let the games begin
# 2021-10-24 15:25:02,234-<run>-[line:94]-INFO: ========================================
# 2021-10-24 15:25:02,235-<run>-[line:95]-INFO: # parameters:
# 2021-10-24 15:25:02,235-<run>-[line:96]-INFO: IMG_H=224
# 2021-10-24 15:25:02,235-<run>-[line:97]-INFO: IMG_W=224
# 2021-10-24 15:25:02,235-<run>-[line:98]-INFO: img_depth=1
# 2021-10-24 15:25:02,235-<run>-[line:99]-INFO: img_depth_out_num=2
# 2021-10-24 15:25:02,235-<run>-[line:100]-INFO: batch_size=16
# 2021-10-24 15:25:02,235-<run>-[line:101]-INFO: n_classes=2
# 2021-10-24 15:25:02,235-<run>-[line:102]-INFO: learning_rate=0.001
# 2021-10-24 15:25:02,235-<run>-[line:103]-INFO: patience_scheduler=5
# 2021-10-24 15:25:02,235-<run>-[line:104]-INFO: patience_early_stop=10
# 2021-10-24 15:25:02,235-<run>-[line:105]-INFO: ReduceLR_factor=0.2
# 2021-10-24 15:25:02,235-<run>-[line:107]-INFO: ========================================
# ==========================================
# Device: cuda
# Current cuda device: 0
# torch.cuda.device_count()= 1
# ==========================================
# SwinTransformerSys expand initial----depths:[2, 2, 6, 2];depths_decoder:[1, 2, 2, 2];drop_path_rate:0.1;num_classes:2
# 2021-10-24 15:25:02,391-<run>-[line:138]-INFO: ----------------------------------------- load train (133812): C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/train/imgs/
# Parameters: 29878848
# Parameters: 29878848
# predict 100 imgs : 5.24617075920105
# predict 100 imgs : 0:00:05.246171
# predict FPS is      : 19.061522125373823/s
# ------------------------------------------------
# predict 1,000 imgs : 17.09200167655945
# predict 1,000 imgs : 0:00:17.092002
# predict FPS is      : 58.506898075690806/s
# ------------------------------------------------
# predict 10,000 imgs : 138.88150811195374
# predict 10,000 imgs : 0:02:18.881508
# predict FPS is      : 72.00382639810408/s
# ------------------------------------------------
# predict 30,000 imgs : 410.3462679386139
# predict 30,000 imgs : 0:06:50.346268
# predict FPS is      : 73.10898707744036/s
# ------------------------------------------------
# predict 50,000 imgs : 667.6072611808777
# predict 50,000 imgs : 0:11:07.607261
# predict FPS is      : 74.89433220297657/s
# ------------------------------------------------
# predict 100,000 imgs : 1353.657604932785
# predict 100,000 imgs : 0:22:33.657605
# predict FPS is      : 73.87392471744391/s
# ------------------------------------------------
# all:2594.763652563095
# all:0:43:14.763652
#
# Process finished with exit code 0
