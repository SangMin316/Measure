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
from Model.Main_model import MainModel
from Model.Network import *
import torch.nn.functional as F



class OneFoldTrainer:
    def __init__(self, args, fold, config):
        self.args = args
        self.fold = fold
        
        self.cfg = config
        self.tp_cfg = config['training_params']
        self.es_cfg = self.tp_cfg['early_stopping']
        self.pf_cfg = config['feature_pyramid']

        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = '7'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        print('')
        print('[INFO] Config Mode: {}'.format(self.tp_cfg['mode']))
        
        self.train_iter = 0

        if self.tp_cfg['mode'] == 'SCL':
            self.ckpt_path = os.path.join('checkpoints_ft',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['tau']))
            self.ckpt_path_from = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['tau']))

        
        elif self.tp_cfg['mode'] == 'WOH':
            # self.ckpt_path = os.path.join('checkpoints_ft',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['h']),str(self.tp_cfg['lambda']))
            self.ckpt_path = os.path.join('checkpoints_ft',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['h']),str(self.tp_cfg['lambda']))
            self.ckpt_path_from = os.path.join('checkpoints_pr2',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['h']),str(self.tp_cfg['lambda']))
      
        else:
            self.ckpt_path = os.path.join('checkpoints_fp',self.cfg['dataset']['name'], self.cfg['backbone']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )
            self.ckpt_path_from = os.path.join('checkpoints_pr',self.cfg['dataset']['name'], self.tp_cfg['mode'], str(self.tp_cfg['beta']), str(self.tp_cfg['tau']) )



        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self.model = self.build_model()

        self.loader_dict = self.build_dataloader()

        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        self.optimizer = optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.tp_cfg['lr'], weight_decay=self.tp_cfg['weight_decay'])
        self.activate_train_mode()

        # dataset -> mode -> setting 
        self.ckpt_name = 'ckpt_fold-{0:02d}.pth'.format(self.fold)
        self.early_stopping = EarlyStopping(patience=self.es_cfg['patience'], verbose=True, ckpt_path=self.ckpt_path, ckpt_name=self.ckpt_name, mode=self.es_cfg['mode'])


    def build_model(self):
        model = MainModel(self.cfg)
        print('[INFO] Number of params of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
        model = torch.nn.DataParallel(model, device_ids=list(range(len(self.args.gpu.split(",")))))
        Pretrain_model_state = torch.load(os.path.join(self.ckpt_path_from,self.ckpt_name))
        model.load_state_dict(Pretrain_model_state, strict=False)

        model.to(self.device)
        print('[INFO] Model prepared, Device used: {} GPU:{}'.format(self.device, self.args.gpu))
        return model



    def build_dataloader(self):
        # dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': False, 'num_workers': 4*len(self.args.gpu.split(",")), 'pin_memory': True}
        dataloader_args = {'batch_size': self.tp_cfg['batch_size'], 'shuffle': True, 'num_workers': 4, 'pin_memory': False}

        train_dataset = EEGDataLoader(self.cfg, self.fold, set='train',shuffle =True)
        # train_dataset = EEGDataLoader(self.cfg, self.fold, set='train',shuffle =True)

        train_loader = DataLoader(dataset=train_dataset, **dataloader_args)
        val_dataset = EEGDataLoader(self.cfg, self.fold, set='val')
        val_loader = DataLoader(dataset=val_dataset, **dataloader_args)
        
        test_dataset = EEGDataLoader(self.cfg, self.fold, set='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.tp_cfg['batch_size'], shuffle=False, num_workers=4*len(self.args.gpu.split(",")), pin_memory=True)
        print('[INFO] Dataloader prepared')

        return {'train': train_loader, 'val': val_loader, 'test': test_loader}



    def activate_train_mode(self):
        self.model.train()
        if self.tp_cfg['step'] == 'train':
            # print('[INFO] Freeze backone')
            self.model.module.feature.train(False)
            for p in self.model.module.feature.parameters():
                p.requires_grad = False

            # print('[INFO] Unfreeze conv_c5')
            self.model.module.feature.conv_c5.train(True)
            for p in self.model.module.feature.conv_c5.parameters(): p.requires_grad = True
            
            if self.pf_cfg['num_scales'] > 1:
                # print('[INFO] Unfreeze conv_c4')
                self.model.module.feature.conv_c4.train(True)
                for p in self.model.module.feature.conv_c4.parameters(): p.requires_grad = True
                
            if self.pf_cfg['num_scales'] > 2:
                # print('[INFO] Unfreeze conv_c3')
                self.model.module.feature.conv_c3.train(True)
                for p in self.model.module.feature.conv_c3.parameters(): p.requires_grad = True

            # self.model.module.feature.init_layer.train(False)
            # self.model.module.feature.layer1.train(False)
            # self.model.module.feature.layer2.train(False)
            # self.model.module.feature.layer3.train(False)
            # self.model.module.feature.layer4.train(False)
            # for p in self.model.module.feature.init_layer.parameters():
            #     p.requires_grad = False
            # for p in self.model.module.feature.layer1.parameters():
            #     p.requires_grad = False
            # for p in self.model.module.feature.layer2.parameters():
            #     p.requires_grad = False
            # for p in self.model.module.feature.layer3.parameters():
            #     p.requires_grad = False
            # for p in self.model.module.feature.layer4.parameters():
            #     p.requires_grad = False




    def train_one_epoch(self):
        # self.model.train()
        # self.Pretrain_model.train()

        correct, total, train_loss = 0, 0, 0

        for i, (inputs, labels,subject_lables) in enumerate(self.loader_dict['train']):

            loss = 0
            inputs = inputs.to(self.device) # batch, 1, 3000
            labels = labels.view(-1).to(self.device)
            total += labels.size(0)
            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]


            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    

            train_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)

            # _, label = torch.max(one_hot_y, 1)
            correct += (predicted == labels).sum().item()
            self.train_iter += 1

            progress_bar(i, len(self.loader_dict['train']), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss / (i + 1), 100. * correct / total, correct, total))
            
            if self.train_iter % self.tp_cfg['val_period'] == 0:
                print('')
                val_loss = self.evaluate(mode='val')
                self.early_stopping(None, val_loss, self.model)
                self.activate_train_mode()
                if self.early_stopping.early_stop:
                    break

    @torch.no_grad()
    def evaluate(self, mode):
        self.model.eval()
        correct, total, eval_loss = 0, 0, 0
        y_true = np.zeros(0)
        y_pred = np.zeros((0, self.cfg['classifier']['num_classes']))

        for i, (inputs, labels, subject_lables) in enumerate(self.loader_dict[mode]):
            loss = 0
            total += labels.size(0)
            inputs = inputs.to(self.device)
            labels = labels.view(-1).to(self.device)

            outputs = self.model(inputs)
            outputs_sum = torch.zeros_like(outputs[0])

            for j in range(len(outputs)):
                loss += self.criterion(outputs[j], labels)
                outputs_sum += outputs[j]

    
            eval_loss += loss.item()
            predicted = torch.argmax(outputs_sum, 1)
            correct += predicted.eq(labels).sum().item()
            
            y_true = np.concatenate([y_true, labels.cpu().numpy()])
            y_pred = np.concatenate([y_pred, outputs_sum.cpu().numpy()])

            progress_bar(i, len(self.loader_dict[mode]), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (eval_loss / (i + 1), 100. * correct / total, correct, total))

        if mode == 'val':
            return eval_loss
        elif mode == 'test':
            return y_true, y_pred
        else:
            raise NotImplementedError
    
    def run(self):
        for epoch in range(self.tp_cfg['max_epochs']):
            print('\n[INFO] Fold: {}, Epoch: {}'.format(self.fold, epoch))
            self.train_one_epoch()
            if self.early_stopping.early_stop:
                break
        
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_path, self.ckpt_name)))
        y_true, y_pred = self.evaluate(mode='test')
        print('')

        return y_true, y_pred

def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=UserWarning) 

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=124712, help='random seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu id')
    # parser.add_argument('--config', type=str, default = './Config_SleepEDF20_fine.json' ,help='config file path')
    parser.add_argument('--config', type=str, default = './Config_MASS_fine.json' ,help='config file path')
    
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # For reproducibility
    set_random_seed(args.seed, use_cuda=True)

    with open(args.config) as config_file:
        config = json.load(config_file)
    config['name'] = os.path.basename(args.config).replace('.json', '')
    
    Y_true = np.zeros(0)
    Y_pred = np.zeros((0, config['classifier']['num_classes']))

    for fold in range(1, config['dataset']['num_splits'] + 1):

        trainer = OneFoldTrainer(args, fold, config)
        y_true, y_pred = trainer.run()
        Y_true = np.concatenate([Y_true, y_true])
        Y_pred = np.concatenate([Y_pred, y_pred])
    
        summarize_result(config, fold, Y_true, Y_pred)
    
if __name__ == "__main__":
    main()

