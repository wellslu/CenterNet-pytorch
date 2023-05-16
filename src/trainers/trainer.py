from abc import ABCMeta, abstractmethod

import mlconfig
import mlflow
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
import os

from ..metrics import Accuracy, Average, MIOU


class AbstractTrainer(metaclass=ABCMeta):

    @abstractmethod
    def fit(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        raise NotImplementedError


@mlconfig.register
class Trainer(AbstractTrainer):

    def __init__(self, device, model, focal_loss, reg_l1, optimizer, scheduler, train_loader, test_loader, num_epochs):
        self.device = device
        self.model = model
        self.focal_loss = focal_loss
        self.reg_l1 = reg_l1
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.best_loss = 99999
        self.epoch = 1

    def fit(self):
        for self.epoch in trange(self.epoch, self.num_epochs + 1):
            train_loss = self.train()
            test_loss = self.evaluate()
            if test_loss.value < self.best_loss:
                self.best_loss = test_loss.value
                os.remove('./model/best_loss.pt')
                torch.save(self.model, './model/best_loss.pt')
            self.scheduler.step()

            self.save_checkpoint('checkpoint.pth')

            metrics = dict(train_loss=train_loss.value,
                           test_loss=test_loss.value,)
            mlflow.log_metrics(metrics, step=self.epoch)
            format_string = 'Epoch: {}/{}, '.format(self.epoch, self.num_epochs)
            format_string += 'train loss: {}, '.format(train_loss)
            format_string += 'test loss: {}'.format(test_loss)
            tqdm.write(format_string)

    def train(self):
        self.model.train()

        train_loss = Average()

        for image, batch_hm, batch_wh, batch_reg, batch_reg_mask in tqdm(self.train_loader):
            x = image.to(self.device)
            batch_hm = batch_hm.to(self.device)
            batch_wh = batch_wh.to(self.device)
            batch_reg = batch_reg.to(self.device)
            batch_reg_mask = batch_reg_mask.to(self.device)

            hm, wh, offset = self.model(x)
            c_loss          = self.focal_loss(hm, batch_hm)
            wh_loss         = 0.1 * self.reg_l1(wh, batch_wh, batch_reg_mask)
            off_loss        = self.reg_l1(offset, batch_reg, batch_reg_mask)
            loss            = c_loss + wh_loss + off_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss.update(loss.item(), number=x.size(0))

        return train_loss

    def evaluate(self):
        self.model.eval()

        test_loss = Average()

        with torch.no_grad():
            for image, batch_hm, batch_wh, batch_reg, batch_reg_mask in tqdm(self.test_loader):
                x = image.to(self.device)
                batch_hm = batch_hm.to(self.device)
                batch_wh = batch_wh.to(self.device)
                batch_reg = batch_reg.to(self.device)
                batch_reg_mask = batch_reg_mask.to(self.device)

                hm, wh, offset = self.model(x)
                c_loss          = self.focal_loss(hm, batch_hm)
                wh_loss         = 0.1 * self.reg_l1(wh, batch_wh, batch_reg_mask)
                off_loss        = self.reg_l1(offset, batch_reg, batch_reg_mask)
                loss            = c_loss + wh_loss + off_loss

                test_loss.update(loss.item(), number=x.size(0))
        
        return test_loss

    def save_checkpoint(self, f):
        self.model.eval()

        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
        }

        torch.save(checkpoint, f)
        mlflow.log_artifact(f)

    def resume(self, f):
        checkpoint = torch.load(f, map_location=self.device)

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch'] + 1
