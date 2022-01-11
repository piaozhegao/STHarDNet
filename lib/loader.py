from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
#from sklearn.model_selection import train_test_split
import torch


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def preprocess(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j][data[i][j] > 5] = 0
    return data


def check_size(img):
    dim = (512, 512)
    if img.shape != dim:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img
def check_size_target(img):
    dim = (512, 512)
    if img.shape != dim:
        img = np.where(img == 0, img, 1)  ##  img category 변환
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    return img

class CustomDataset(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        batch_arr_x = torch.tensor([check_size(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_x],
                                   dtype=torch.float32)
        batch_arr_x = batch_arr_x.reshape((1, 512, 512))
        batch_arr_y = torch.tensor([check_size_target(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_y],
                                   dtype=torch.float32)
        batch_arr_y = batch_arr_y.reshape((1, 512, 512))
        batch_arr_y = preprocess(batch_arr_y)

        return batch_arr_x, batch_arr_y

class CustomDataset_color(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        batch_arr_x = torch.squeeze(torch.tensor([check_size2(cv2.imread(name)) for name in batch_x], dtype=torch.float32), 0)
        batch_arr_x = batch_arr_x.permute(2, 0, 1)  ## channel 위치 변경

        batch_arr_y = torch.tensor([check_size(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_y], dtype=torch.float32)
        batch_arr_y = batch_arr_y.reshape((1, 512, 512))
        batch_arr_y = preprocess(batch_arr_y)

        return batch_arr_x, batch_arr_y



def check_size2(img):
    dim = (224, 224)
    if img.shape != dim:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img
def check_size2_target(img):
    dim = (224, 224)
    if img.shape != dim:
        img = np.where(img == 0, img, 1)  ##  img category 변환
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    return img

class CustomDataset2(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        ## load color  img
        batch_arr_x = torch.squeeze(torch.tensor([check_size2(cv2.imread(name)) for name in batch_x], dtype=torch.float32), 0)
        batch_arr_x = batch_arr_x.permute(2, 0, 1)  ## channel 위치 변경

        batch_arr_y = torch.squeeze(torch.tensor([check_size2_target(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_y], dtype=torch.float32), 0)
        batch_arr_y = batch_arr_y.reshape((1, 224, 224))
        batch_arr_y = preprocess(batch_arr_y)
        return batch_arr_x, batch_arr_y

class CustomDataset2_gray(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        ## load gray img
        batch_arr_x = torch.squeeze(torch.tensor([check_size2(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_x], dtype=torch.float32), 0)
        # batch_arr_x = batch_arr_x.permute(2, 0, 1)  ## channel 위치 변경
        batch_arr_x = batch_arr_x.reshape((1, 224, 224))

        batch_arr_y = torch.squeeze( torch.tensor([check_size2_target(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_y], dtype=torch.float32), 0)
        batch_arr_y = batch_arr_y.reshape((1, 224, 224))
        batch_arr_y = preprocess(batch_arr_y)

        # ##  load color image
        # batch_arr_x = torch.tensor([check_size2(cv2.imread(name)) for name in batch_x],
        #                            dtype=torch.float32)
        #
        # batch_arr_x = batch_arr_x.reshape((3, 224, 224))
        # batch_arr_y = torch.tensor([check_size2(cv2.imread(name, cv2.IMREAD_GRAYSCALE)) for name in batch_y],
        #                            dtype=torch.float32)
        # batch_arr_y = batch_arr_y.reshape((1, 224, 224))
        # batch_arr_y = preprocess(batch_arr_y)

        return batch_arr_x, batch_arr_y




def check_size_resize(img, is_target=True):
    if is_target:
        dim = (224, 224)
        if img.shape != dim:
            img = np.where(img == 0, img, 1)  ##  img category 변환
            img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    else:
        #dim = (512, 512)
        dim = (256, 256)
        if img.shape != dim:
            img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img
class CustomDataset_resize(Dataset):
    def __init__(self, x_set, y_set):
        self.x, self.y = x_set, y_set

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        batch_arr_x = torch.tensor([check_size_resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), is_target=False) for name in batch_x], dtype=torch.float32)
        #batch_arr_x = batch_arr_x.reshape((1, 512, 512))
        batch_arr_x = batch_arr_x.reshape((1, 256, 256))
        batch_arr_y = torch.tensor([check_size_resize(cv2.imread(name, cv2.IMREAD_GRAYSCALE), is_target=True) for name in batch_y], dtype=torch.float32)
        batch_arr_y = batch_arr_y.reshape((1, 224, 224))
        batch_arr_y = preprocess(batch_arr_y)

        return batch_arr_x, batch_arr_y




###################################
def check_size_3d(img, img_w=192, img_h=192):
    dim = (img_w, img_h)
    if img.shape != dim:
        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img
def check_size_3d_target(img, img_w=192, img_h=192):
    dim = (img_w, img_h)
    if img.shape != dim:
        img = np.where(img == 0, img, 1)  ##  img category 변환
        img = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)
    return img

class CustomDataset_3d_gray(Dataset):
    def __init__(self, x_set, y_set, img_w, img_h):
        self.x, self.y = x_set, y_set
        self.img_w = img_w
        self.img_h = img_h

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        ##
        #['C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png']
        #for names in batch_x:
        i = 0
        batch_arr_x = torch.tensor([0])
        for name in batch_x[0]:
            # logging.debug("name is : {}".format(name))
            img = check_size_3d(cv2.imread(name, cv2.IMREAD_GRAYSCALE), img_w=self.img_w, img_h=self.img_h)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.reshape((1, self.img_w, self.img_h))
            if i == 0:
                batch_arr_x = img
            else:
                batch_arr_x = torch.cat([batch_arr_x, img], dim=0)
            i += 1
        batch_arr_x = torch.squeeze(batch_arr_x, 0)
        # logging.info("batch_arr_x:{}".format(batch_arr_x.shape))

        i = 0
        batch_arr_y = []
        for name in batch_y[0]:
            # logging.debug("name is : {}".format(name))
            img = check_size_3d_target(cv2.imread(name, cv2.IMREAD_GRAYSCALE), img_w=self.img_w, img_h=self.img_h)
            # logging.debug("img.shape={}".format(img.shape))
            if i == 0:
                batch_arr_y = img
            else:
                batch_arr_y = batch_arr_y | img
            i += 1
            # logging.debug("batch_arr_y.sum = {}".format(np.sum(batch_arr_y)))
        # logging.debug("batch_arr_y {} ".format(batch_arr_y.shape))
        batch_arr_y = torch.squeeze(torch.tensor(batch_arr_y, dtype=torch.float32), 0)
        batch_arr_y = batch_arr_y.reshape((1, self.img_w, self.img_h))
        # logging.debug("batch_arr_y after reshape: {} ".format(batch_arr_y.shape))
        return batch_arr_x, batch_arr_y



def get_3d_img_list(path, depth):
    names_3d_list = []
    img_depth = depth
    for name in os.listdir(path):
        # print (val_img_path)
        name_tmps = name.split('_')
        name_id = name.split('_')[3].split(".")[0]
        name_id = int(name_id)
        #name_id = name_id+105  # for test

        # for i in range(189):
        #     num_i = name_id + i
        num_i = name_id
        if num_i <= 189 - img_depth + 1:
            names_3d = []
            for d in range(img_depth):
                #name_3d = name_tmps[0] + "_" + name_tmps[1] + "_" + name_tmps[2] + "_{:03d}".format(name_id+i+d) + ".png"
                name_3d = name_tmps[0] + "_" + name_tmps[1] + "_" + name_tmps[2] + "_{:03d}".format(name_id + d) + ".png"
                names_3d.append(os.path.join(path, name_3d))
            names_3d_list.append(names_3d)
        #break
    return names_3d_list





class CustomDataset_3d_gray_out_1(Dataset):
    def __init__(self, x_set, y_set, img_w, img_h, out_num=2):
        self.x, self.y = x_set, y_set
        self.img_w = img_w
        self.img_h = img_h
        self.out_num = out_num

    def __len__(self):
        # returns the number of batches
        return len(self.x)

    def __getitem__(self, idx):
        # returns one batch
        ...
        batch_x = self.x[idx:idx + 1]
        batch_y = self.y[idx:idx + 1]

        ##
        #['C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png',
        # 'C:/Users/user/Documents/python-code/11_c_paper/data/ATLAS/by_id/val/imgs/Site1_031770_t01_001.png']
        #for names in batch_x:
        i = 0
        batch_arr_x = torch.tensor([0])
        for name in batch_x[0]:
            # logging.debug("name is : {}".format(name))
            img = check_size_3d(cv2.imread(name, cv2.IMREAD_GRAYSCALE), img_w=self.img_w, img_h=self.img_h)
            img = torch.tensor(img, dtype=torch.float32)
            img = img.reshape((1, self.img_w, self.img_h))
            if i == 0:
                batch_arr_x = img
            else:
                batch_arr_x = torch.cat([batch_arr_x, img], dim=0)
            i += 1
        batch_arr_x = torch.squeeze(batch_arr_x, 0)
        # logging.info("batch_arr_x:{}".format(batch_arr_x.shape))

        i = 0
        batch_arr_y = []
        for name in batch_y[0]:
            # logging.debug("name is : {}".format(name))
            img = check_size_3d_target(cv2.imread(name, cv2.IMREAD_GRAYSCALE), img_w=self.img_w, img_h=self.img_h)
            # logging.debug("img.shape={}".format(img.shape))
            if i == self.out_num:  ##  0,1,2,3
                batch_arr_y = img
            # if i == 0:
            #     batch_arr_y = img
            # else:
            #     batch_arr_y = batch_arr_y | img
            i += 1
            # logging.debug("batch_arr_y.sum = {}".format(np.sum(batch_arr_y)))
        # logging.debug("batch_arr_y {} ".format(batch_arr_y.shape))
        batch_arr_y = torch.squeeze(torch.tensor(batch_arr_y, dtype=torch.float32), 0)
        batch_arr_y = batch_arr_y.reshape((1, self.img_w, self.img_h))
        # logging.debug("batch_arr_y after reshape: {} ".format(batch_arr_y.shape))
        return batch_arr_x, batch_arr_y










if __name__ == "__main__":
    batch_size = 4

    train_img_path = '../data/0223_split/train/imgs/'
    train_mask_path = '../data/0223_split/train/masks/'
    val_img_path = '../data/0223_split/val/imgs/'
    val_mask_path = '../data/0223_split/val/masks/'

    train_data_list = [train_img_path + i for i in os.listdir(train_img_path)]
    train_mask_list = [train_mask_path + i for i in os.listdir(train_mask_path)]
    val_data_list = [val_img_path + i for i in os.listdir(val_img_path)]
    val_mask_list = [val_mask_path + i for i in os.listdir(val_mask_path)]

    train_data = DataLoader(CustomDataset(train_data_list, train_mask_list), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(CustomDataset(val_data_list, val_mask_list), batch_size=batch_size, shuffle=True)
    data = iter(train_data)
    img, label = data.next()
    print(img.shape, label.shape, len(train_data), len(val_data))
