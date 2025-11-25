import os

import json
import argparse
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from loss import *
from loader import EEGDataLoader
from Model.Network import *
import torch.nn.functional as F

from Model.Main_model import MainModel


class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        print('')
        print('[INFO] Config Mode: {}'.format(self.tp_cfg['mode']))
        
        self.train_iter = 0
        self.model = self.build_model()
        self.loader_dict = self.build_dataloader()
        if self.tp_cfg['mode'] == 'SCL':
            self.criterion = SCL(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['h']) )

        elif self.tp_cfg['mode'] == 'CL':
            self.criterion = CL(temperature=self.tp_cfg['temperature'], seq_len  = self.cfg['dataset']['seq_len'], device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'],self.cfg['backbone']['name'], self.tp_cfg['mode'], str(self.tp_cfg['h']) )

        elif self.tp_cfg['mode'] == 'Ours_HML':
            self.criterion = Ours_HML(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )

    
        elif self.tp_cfg['mode'] == 'Ours_HL':
            self.criterion = Ours_HL(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )
    
        elif self.tp_cfg['mode'] == 'Ours_HM':
            self.criterion = Ours_HM(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )

        elif self.tp_cfg['mode'] == 'PCL':
            self.criterion = PCL()
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.cfg['backbone']['name'],self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )

        elif self.tp_cfg['mode'] == 'Ours_WOH':
            self.criterion = Ours_WOH(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )


        elif self.tp_cfg['mode'] == 'MVEB_sup':
            self.criterion = MVEB_sup(config= self.cfg, device = self.device)
            self.ckpt_path = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        
        # dataset -> mode -> setting 
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])



    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))

        return model
    
    def build_dataloader(self):
        # dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': False, 'num_workers': 4*len(self.args.gpu.split(",")), 'pin_memory': True}
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': False, 'num_workers': 4, 'pin_memory': False}

        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train',shuffle =True)
        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader}



    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        align_loss = 0
        entropy_loss = 0
        for i, (inputs, labels,subject_lables) in enumerate(self.loader_dict['train']):
            labels = labels.to(self.device)

            inputs = torch.cat([inputs[0].to(self.device),inputs[1].to(self.device)])

            subject_lables = subject_lables.to(self.device)
            loss, align, entropy = self.criterion.forward(inputs, labels,subject_lables,self.model)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            align_loss += align.item()
            entropy_loss += entropy.item()

            self.train_iter += 1
  
            progress_bar(i, len(self.loader_dict['train']), 'Lr: %.4e | Loss: %.3f | Align: %.3f | Entropy: %.3f' %(get_lr(self.optimizer), train_loss / (i + 1), align_loss / (i + 1),entropy_loss / (i + 1)))

            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.model.train()
                if self.early_stopping.early_stop:
                    break

    #
    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        eval_loss = 0
        align_loss = 0
        entropy_loss = 0
        for i, (inputs, labels, subject_lables) in enumerate(self.loader_dict[mode]):
            labels = labels.to(self.device)
            inputs = torch.cat([inputs[0].to(self.device),inputs[1].to(self.device)])
            subject_lables = subject_lables.to(self.device)
            loss, align, entropy = self.criterion.forward(inputs, labels,subject_lables,self.model)
            align_loss += align.item()
            entropy_loss += entropy.item()
        
            eval_loss += loss.item()
        
            progress_bar(i, len(self.loader_dict[mode]), 'Lr: %.4e | Loss: %.3f | Align: %.3f | Entropy: %.3f' %(get_lr(self.optimizer), eval_loss / (i + 1), align_loss / (i + 1),entropy_loss / (i + 1)))

        return eval_loss
    
    def test(self,mode):
        self.model.eval()
        labels_list = []
        y_pred_list = []
        for i, (inputs, labels, subject_lables) in enumerate(self.loader_dict[mode]):
            labels = labels.to(self.device)
            inputs = torch.cat([inputs[0].to(self.device),inputs[1].to(self.device)])
            labels = labels.view(-1,1)

            
            y_pred = self.model.forward(inputs)
            labels_list.extend(labels.numpy())
            y_pred_list.extend(y_pred.detach().cpu().numpy())



        return np.array(labels_list), np.array(y_pred_list)



    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=316, help='random seed')
    parser.add_argument('--gpu', type=str, default="2", help='gpu id')
    parser.add_argument('--config', type=str, default = './Config_SleepEDF20_pre.json' ,help='config file path')
    # parser.add_argument('--config', type=str, default = './Config_MASS_pre.json' ,help='config file path')

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    
    for fold in range(1, config['dataset']['num_splits'] + 1):

        trainer = OneFoldTrainer(args, fold, config)
        trainer.run()


if __name__ == "__main__":
    main()

