import os
import torch

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Config:
    '''Configuration file'''
    def __init__(self):
        super().__init__()
        #image_parameters
        self.input_image_h = 384
        self.input_image_w = 960
        self.input_image_c = 3
        #file_parameters
        if not os.path.exists('weights'):
            os.makedirs('weights')
        if not os.path.exists('lists'):
            os.makedirs('lists')
        if os.path.exists(os.path.join('lists', 'train.txt')) and\
            os.path.exists(os.path.join('lists', 'val.txt')) and\
            os.path.exists(os.path.join('lists', 'test.txt')):
            self.train_dir = os.path.join('lists', 'train.txt')
            self.val_dir = os.path.join('lists', 'val.txt')
            self.test_dir = os.path.join('lists', 'test.txt')
            with open(self.train_dir, "r") as f:
                self.num_train_images = len(f.readlines())
            with open(self.val_dir, "r") as f:
                self.num_val_images = len(f.readlines())
            with open(self.test_dir, "r") as f:
                self.num_test_images = len(f.readlines())
        else:
            print('train.txt or val.txt or test.txt don`t exist, create it using utils.create_train_val_test_lists')
        #model_parameters
        self.device = 'cuda:0'
        self.pretrained_weights = None
        self.initial_epoch = 0 if self.pretrained_weights == None\
            else int(self.pretrained_weights[-6:-3]) + 1
        self.class_weight = [0.01, 0.6]
        self.num_workers = 4
        self.learning_rate = 0.001
        self.n_classes = 2
        self.mean = [0.33677723, 0.36253921, 0.38315998]
        self.std = [0.26173086, 0.27664054, 0.27986822]
        self.batch_size = 8
        self.num_epochs = 100
        #layers_parameters
        self.drop_rate = 0.4