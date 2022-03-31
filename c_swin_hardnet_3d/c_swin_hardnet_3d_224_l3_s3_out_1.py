

import logging
from torch import nn
from torch.optim import Adam
from loss import DiceCE, Dice_IoULoss
import torch
from torch.utils.data import DataLoader
import os
from loader import CustomDataset_3d_gray_out_1
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import get_logger
from tqdm import tqdm
from pytorchtools import EarlyStopping
from c_metrics import runningScore, averageMeter
import datetime
from swin_hardnet import swin_hardnet_all_transformer_embed_dim_3d
from c_metrics import get_score_from_all_slices_cherhoo


import logging
logging.basicConfig(format='%(asctime)s-<%(funcName)s>-[line:%(lineno)d]-%(levelname)s: %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # define logging print level
logger.setLevel(logging.INFO)  # define logging print level


# import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8,9"  # Set the GPUs 2 and 3 to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"  # Set the GPUs 2 and 3 to use
os.environ["CUDA_VISIBLE_DEVICES"] = "8"  # Set the GPUs 2 and 3 to use












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
        if num_i <= 189 - img_depth + 1:
            names_3d = []
            for d in range(img_depth):
                name_3d = name_tmps[0] + "_" + name_tmps[1] + "_" + name_tmps[2] + "_{:03d}".format(name_id + d) + ".png"
                names_3d.append(os.path.join(path, name_3d))
            names_3d_list.append(names_3d)
        #break
    return names_3d_list



import numpy as np
test_path = 'D:/cherhoo/datas/ATLAS/by_id/'
categories = os.listdir(test_path)
test_datas = {}
ctg = 'val'
data_path = os.path.join(test_path, ctg) + '/imgs/'
mask_path = os.path.join(test_path, ctg) + '/masks/'
val_data_list = get_3d_img_list(data_path, img_depth)
val_mask_list = get_3d_img_list(mask_path, img_depth)
test_datagen = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W), batch_size=8, shuffle=False, num_workers=4)
test_datas[ctg] = test_datagen
################################################


def get_labels():
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = []
    for i, pack in enumerate(test_datagen):
        images, labels = pack
        labels = labels.to(device_).long()
        labels = labels.cpu().detach().numpy()
        labels_ = []
        for ii in range(len(labels)):
            label = labels[ii].transpose(1, 2, 0)
            labels_.append(label)
        labels_ = torch.tensor(labels_)
        if i == 0:
            label_list = labels_
        else:
            label_list = torch.cat([label_list, labels_], 0)

    return label_list


def convert_label_224(outputs):
    pred = np.zeros((224, 224, 1))
    for i in range(224):
        pred[i] = np.argmax(outputs[i], axis=-1).reshape((-1, 1))
    return pred

def test(model, device, test_datas, label_list_global):
    mean_ctg = {}
    num = 1

    output_list = []
    for i, pack in enumerate(test_datagen):
        images, labels = pack
        images = images.to(device)
        outputs = model(images)
        outputs = outputs.cpu().detach().numpy()
        outputs_ = []
        for ii in range(len(outputs)):
            output = outputs[ii].transpose(1, 2, 0)
            output = convert_label_224(output)
            outputs_.append(output)
        outputs_ = torch.tensor(outputs_)

        if i == 0:
            output_list = outputs_
        else:
            output_list = torch.cat([output_list, outputs_], 0)
    scores = get_score_from_all_slices_cherhoo(labels=label_list_global, predicts=output_list)

    mean_score = {}
    for key in scores.keys():
        mean_score[key] = np.mean(scores[key])
        if key not in mean_ctg.keys():
            mean_ctg[key] = mean_score[key]
        else:
            mean_ctg[key] += mean_score[key]

    ##################################
    json_all = {
        "scores": scores,
        "mean_score": mean_score
    }
    logging.info(mean_score)
    return str(json_all['mean_score'])














def train():
    batch_size = 8*2 *2 + 16 +16  #+8+16
    n_classes = 2
    # learning_rate = 0.01
    learning_rate = 0.001
    model_name = "c_swin_hardnet_3d_224_l3_s3_ATLAS"  ##  c_swin_cnn_unet
    patience_scheduler = 5*2
    patience_early_stop = 15*2
    ReduceLR_factor = 0.5

    start_iter = 0
    num_epochs = 500

    ## save model log and check points
    logdir = './runs/exp1_'+model_name+'_D0929_dicece_0.001'+'_out_1-1'+"_20211031"
    writer = SummaryWriter(log_dir=logdir)
    logger = get_logger(logdir)
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
    logger.info("logdir={}".format(logdir))
    logger.info("========================================")

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

    train_img_path = 'D:/cherhoo/datas/ATLAS/by_id/train/imgs/'
    train_mask_path = 'D:/cherhoo/datas/ATLAS/by_id/train/masks/'
    val_img_path = 'D:/cherhoo/datas/ATLAS/by_id/val/imgs/'
    val_mask_path = 'D:/cherhoo/datas/ATLAS/by_id/val/masks/'

    train_data_list = get_3d_img_list(train_img_path, img_depth)
    train_mask_list = get_3d_img_list(train_mask_path, img_depth)
    val_data_list = get_3d_img_list(val_img_path, img_depth)
    val_mask_list = get_3d_img_list(val_mask_path, img_depth)

    train_data = DataLoader(CustomDataset_3d_gray_out_1(train_data_list, train_mask_list, img_h=IMG_H, img_w=IMG_W), batch_size=batch_size, shuffle=True, num_workers=4*2)
    val_data   = DataLoader(CustomDataset_3d_gray_out_1(val_data_list, val_mask_list, img_h=IMG_H, img_w=IMG_W), batch_size=batch_size, shuffle=True, num_workers=4*2)
    label_list_global = get_labels()

    running_metrics_val = runningScore(n_classes)
    criterion = DiceCE()
    # criterion = Dice_IoULoss()
    # criterion = DiceCELoss()
    criterion.to(device)

    ## create model
    model = swin_hardnet_all_transformer_embed_dim_3d(n_classes=2, in_channels=img_depth, st_img_size=112, st_windows=7).cuda()
    ##  load pre-trained model
    # model_path = "./runs/exp2_swin_hardnet_v2_dicece_0.01/swin_hardnet_v2_epoch_30_checkpoint.pkl"

    # model_path = os.path.join(logdir, model_name+"_best_model.pkl")
    # model.load_state_dict(torch.load(model_path)["model_state"])
    # #@model.eval()
    # model.to(device)


    # model = nn.DataParallel(model)
    total_params = sum(p.numel() for p in model.parameters())
    print('Parameters:', total_params)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=ReduceLR_factor, patience=patience_scheduler, verbose=True)

    best_iou = -100.0
    loss_all = 0
    loss_n = 0
    flag = True
    val_loss_meter = averageMeter()
    train_loss_meter = averageMeter()
    time_meter = averageMeter()


    early_stopping = EarlyStopping(patience=patience_early_stop, verbose=True)
    val_loss_for_print = 0
    while start_iter <= num_epochs and flag:
        for i, pack in enumerate(train_data):
            i += 1
            model.train()
            images, labels = pack
            images = images.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels.squeeze(1))
            # scheduler.step()
            loss.backward()
            optimizer.step()

            loss_all += loss.item()
            train_loss_meter.update(loss.item())  ## train loss update, 모니터링 용
            loss_n += 1

            if (i + 1) % print_interval == 0:
                fmt_str = "in epoch[{}]  Iter [{:d}/{:d}]  Loss: {:.8f} ;  val_loss:{:.8f}"
                print_str = fmt_str.format(start_iter, i + 1, len(train_data), loss_all / loss_n, val_loss_for_print)

                #print(print_str)
                #print(datetime.datetime.now())
                logger.info(print_str)
                # writer.add_scalar("loss/train_loss", loss.item(), i + 1)

        #torch.cuda.empty_cache()
        model.eval()
        loss_all = 0
        loss_n = 0
        with torch.no_grad():
            for i_val, (images_val, labels_val) in tqdm(enumerate(val_data)):
                images_val = images_val.to(device)
                labels_val = labels_val.to(device).long().squeeze(1)

                outputs = model(images_val)
                val_loss = criterion(outputs, labels_val)

                pred = outputs.data.max(1)[1].cpu().numpy()
                gt = labels_val.data.cpu().numpy()

                running_metrics_val.update(gt, pred)
                val_loss_meter.update(val_loss.item())

        writer.add_scalar("loss/val_loss", val_loss_meter.avg, start_iter + 1)
        logger.info("Iter %d Val Loss: %.8f" % (start_iter + 1, val_loss_meter.avg))

        ##  early stopping 및 learning 조정
        my_total_loss = train_loss_meter.avg + val_loss_meter.avg
        val_loss_for_print = val_loss_meter.avg
        logger.info("ecpch[{}] loss: {:.8f} ;  val_loss:{:.8f} my_total_loss:{:.8f}".format(start_iter, train_loss_meter.avg, val_loss_meter.avg, my_total_loss))
        scheduler.step(val_loss_meter.avg)
        early_stopping(val_loss_meter.avg, model)

        if early_stopping.early_stop:
            print("Early stopping : {}".format(datetime.datetime.now()))
            break

        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            logger.info("score {}: {}".format(k, v))
            writer.add_scalar("val_metrics/{}".format(k), v, start_iter + 1)

        for k, v in class_iou.items():
            logger.info("start_iter:{}  class_iou: {}: {}".format(start_iter, k, v))
            writer.add_scalar("val_metrics/cls_{}".format(k), v, start_iter + 1)

        val_loss_meter.reset()
        train_loss_meter.reset()
        running_metrics_val.reset()

        save_path = os.path.join(writer.file_writer.get_logdir(), "{}_{}_checkpoint.pkl".format(model_name, 'epoch_' + str(start_iter + 1)))
        state = {
            "epoch": start_iter + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }
        torch.save(state, save_path)

        if score["Mean IoU : \t"] >= best_iou:
            best_iou = score["Mean IoU : \t"]
            state = {
                "epoch": start_iter + 1,
                "model_state": model.state_dict(),
                "best_iou": best_iou,
            }
            save_path = os.path.join(writer.file_writer.get_logdir(),"{}_best_model.pkl".format(model_name))
            torch.save(state, save_path)
        if start_iter > 200:
            test_score = test(model, device, test_datas, label_list_global)
            logger.info("============================")
            logger.info("----------- in test dataset score in epoch:{} +1 -----------------".format(start_iter))
            logger.info(test_score)
            logger.info("============================\n\n")

        torch.cuda.empty_cache()

        start_iter += 1
    return model, device, test_datas, label_list_global, logdir, model_name, logger



if __name__ == '__main__':
    torch.cuda.empty_cache()
    model, device, test_datas, label_list_global, logdir, model_name, logger_ = train()

    ## print score in best model
    model_path = os.path.join(logdir, model_name + "_best_model.pkl")
    model.load_state_dict(torch.load(model_path)["model_state"])
    model.eval()
    model.to(device)
    test_score = test(model, device, test_datas, label_list_global)
    logger_.info("============================")
    logger_.info("=============  in best model ===============")
    logger_.info(test_score)
    logger_.info("============================\n\n")



