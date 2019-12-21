import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from parts import Metrics
import config
from model import generate_model
from utils import ImageGenerator, ImageGeneratorTest
from multiprocessing import set_start_method

np.random.seed(42)
set_start_method('spawn', True)
cfg = config.Config()


class TrainModel:

    def __init__(self, model):

        self.model = model
        self.model.to(cfg.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate, weight_decay=1e-3)
        class_weight = torch.Tensor(cfg.class_weight)
        self.criterion = nn.CrossEntropyLoss(weight=class_weight).to(cfg.device)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, [1, 3, 7], gamma=0.5)
        self.writer = SummaryWriter('logs/{}_lr={:06}_bz={}_clsW={}'\
            .format(time.time(), cfg.learning_rate, cfg.batch_size, cfg.class_weight))
        # self.write_graph()
        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.train_dir, num_images=cfg.num_train_images),
            batch_size=cfg.batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.val_dir, num_images=cfg.num_val_images),
            batch_size=cfg.batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(
            ImageGenerator(input_path=cfg.test_dir, num_images=cfg.num_test_images),
            batch_size=1)
        }
        self.global_train_step = 0

    def write_graph(self):
        data = torch.zeros((cfg.batch_size, cfg.input_image_c,\
                            cfg.input_image_h, cfg.input_image_w)).to(cfg.device)
        self.writer.add_graph(self.model, data)

    def write_tensorboard_logs(self, phase, loss, log_step, label, pred, acc):
        self.writer.add_scalar('{} loss'.format(phase),
                                loss,
                                log_step)
        self.writer.add_scalar('{}_balanced_accuracy'.format(phase),
                                Metrics.balanced_accuracy(label, pred),
                                log_step)
        self.writer.add_scalar('{}_accuracy'.format(phase),
                                acc,
                                log_step)
        self.writer.add_scalar('{}_compute_tpr'.format(phase),
                                Metrics.compute_tpr(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_tnr'.format(phase),
                                Metrics.compute_tnr(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_ppv'.format(phase),
                                Metrics.compute_ppv(label, pred),
                                log_step)
        self.writer.add_scalar('{}_compute_npv'.format(phase),
                                Metrics.compute_npv(label, pred),
                                log_step)
    
    def write_tensorboard_images(self, pred, label, phase, log_step):
        pred = pred.to('cpu').numpy()
        label = torch.unsqueeze(label, 1).to('cpu').numpy()
        imgs = np.concatenate((label, np.zeros((8-label.shape[0], label.shape[1], label.shape[2], label.shape[3])),\
            pred))

        self.writer.add_images('Label vs Pred, phase: {}'.format(phase), imgs,\
            log_step, dataformats='NCHW')

    def train(self, epoch):
        self.model.train()
        running_loss = 0.
        for i, sample in enumerate(self.dataloaders['train'], 0):
            start_time = time.time()
            inputs = sample['data'].to(cfg.device)
            label = sample['label'].type(torch.LongTensor).to(cfg.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, label)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 0.8)
            self.optimizer.step()
            running_loss += loss.item()

            pred = outputs.max(1, keepdim=True)[1]
            correct = pred.eq(label.view_as(pred)).sum().item()
            acc = int(correct) / (cfg.batch_size * cfg.input_image_h * cfg.input_image_w)

            time_train_batch = time.time() - start_time
            print("Phase: {}, Epoch: {:02}, Iter: {:05}, Loss: {:.5f}, Accuracy: {:.3f}, Time: {:.0f}m {:.0f}s"\
                .format('Train', epoch, i, loss.item(), acc, time_train_batch // 60, time_train_batch % 60))
            if i == 0:
                self.write_tensorboard_logs(phase='Train', loss=running_loss, log_step=self.global_train_step,\
                    label=label, pred=pred, acc=acc)
                self.write_tensorboard_images(pred, label, 'Train', self.global_train_step)
                self.global_train_step += 1
            elif i % 100 == 0:
                running_loss /= 100
                self.write_tensorboard_logs(phase='Train', loss=running_loss, log_step=self.global_train_step,\
                    label=label, pred=pred, acc=acc)
                self.write_tensorboard_images(pred, label, 'Train', self.global_train_step)
                running_loss = 0.
                self.global_train_step += 1

    def val(self, epoch):
        self.model.eval()
        running_loss = 0.
        for i, sample in enumerate(self.dataloaders['val'], 0):
            start_time = time.time()
            inputs = sample['data'].to(cfg.device)
            label = sample['label'].type(torch.LongTensor).to(cfg.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, label)

            pred = outputs.max(1, keepdim=True)[1]
            correct = pred.eq(label.view_as(pred)).sum().item()
            acc = int(correct) / (cfg.batch_size * cfg.input_image_h * cfg.input_image_w)
            running_loss += loss.item()

            time_validation_batch = time.time() - start_time
            print("Phase: {}, Epoch: {:02}, Iter: {:05}, Loss: {:.5f}, Accuracy: {:.3f}, Time: {:.0f}m {:.0f}s"\
                .format('Val', epoch, i, loss.item(), acc, time_validation_batch // 60, time_validation_batch % 60))

        running_loss /= len(self.dataloaders['val'])
        self.write_tensorboard_logs(phase='Validation', loss=loss.item(), log_step=epoch,\
            label=label, pred=pred, acc=acc)
        self.write_tensorboard_images(pred, label, 'Validation', epoch)
    
    def test(self):
        self.model.eval()
        for i, sample in enumerate(self.dataloaders['test'], 0):
            inputs = sample['data'].to(cfg.device)
            start_time = time.time()
            outputs = self.model(inputs)
            time_test_batch = time.time() - start_time
            pred = outputs.max(1, keepdim=True)[1]
            
            plt.figure(1)
            plt.imshow(torch.squeeze(pred).to('cpu').numpy())
            plt.figure(2)
            plt.imshow(torch.squeeze(sample['label']).to('cpu').numpy())
            plt.show()
            print("Number: {}, Time: {}".format(i, time_test_batch))
    
    def test_weights(self):
        num_weights = os.listdir('weights')
        for i_weights in range(len(num_weights)):
            path_weight = os.path.join('weights', 'epoch_{:03}.pt'.format(i_weights))
            self.model = generate_model(pretrained_weights=path_weight)
            self.model.to(cfg.device)
            self.model.eval()
            
            sample = next(iter(self.dataloaders['test']))
            inputs = sample['data'].to(cfg.device)
            outputs = self.model(inputs)
            pred = outputs.max(1)[1]

            img_pred = torch.squeeze(pred).to('cpu').numpy()
            img_label = torch.squeeze(sample['label']).to('cpu').numpy()
            
            fig, axs = plt.subplots(nrows=1, ncols=2)
            ax = axs[0]
            ax.imshow(img_pred)
            ax.set_title('prediction')
            ax = axs[1]
            ax.imshow(img_label)
            ax.set_title('label')
            fig.savefig(os.path.join('weights', 'Figure_{:03}.png'.format(i_weights)))
            plt.close()

    def create_test_videos(self):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_pred = cv2.VideoWriter('predicted.avi', fourcc, 20.0, (cfg.input_image_w, cfg.input_image_h))
        video_label = cv2.VideoWriter('labeled.avi', fourcc, 20.0, (cfg.input_image_w, cfg.input_image_h))

        with open(os.path.join('lists', 'video_list.txt'), 'r') as f:
            img_list = f.readlines()

        test_dataloader = torch.utils.data.DataLoader(ImageGeneratorTest(\
            input_path=os.path.join('lists','video_list.txt'), num_images=len(img_list)))
        self.model.eval()

        for i, sample in enumerate(test_dataloader, 0):
                    
            inputs = sample['data'].to(cfg.device)
            labels = sample['label'].type(torch.LongTensor).to(cfg.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            pred = torch.squeeze(outputs.max(1, keepdim=True)[1])

            label_image = np.expand_dims(torch.squeeze(sample['label']).to('cpu').numpy(), 2).astype(np.uint8)
            label_image = np.concatenate((label_image*255, np.zeros_like(label_image, dtype=np.uint8), np.zeros_like(label_image, dtype=np.uint8)), axis=2)

            pred_image = np.expand_dims(pred.to('cpu').numpy(), 2).astype(np.uint8)
            pred_image = np.concatenate((pred_image*255, np.zeros_like(pred_image, dtype=np.uint8), np.zeros_like(pred_image, dtype=np.uint8)), axis=2)

            input_image = np.squeeze(sample['raw_image'].numpy().astype(np.uint8))

            label_image = cv2.add(input_image, label_image)
            pred_image = cv2.add(input_image, pred_image)

            plt.figure(1)
            plt.imshow(pred_image)
            plt.figure(2)
            plt.imshow(label_image)
            plt.show()

            video_pred.write(pred_image)
            video_label.write(label_image)

            print("Number: {}".format(i))
        
        video_pred.release()
        video_label.release()

    def main(self):
        for epoch in range(cfg.initial_epoch, cfg.num_epochs):
            self.train(epoch)
            self.val(epoch)
            self.scheduler.step(epoch=epoch)
            torch.save(self.model.state_dict(), os.path.join('weights', 'epoch_{:03}.pt'.format(epoch)))


if __name__ == '__main__':
    model = generate_model()
    trainer = TrainModel(model)
    trainer.main()
    # trainer.test()
    # trainer.create_test_videos()
    # trainer.test_weights()
