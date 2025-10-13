import torch.nn as nn
import torch.nn.functional as F
import torch

from .xsleepnet import XSleepNetFeature


from .classifiers import get_classifier
from .sleepyco_multi import SleePyCoBackbone
from .ResNet import FeatureCNN


last_chn_dict = {
    'SleePyCo': 256,
    'XSleepNet': 256,
    'IITNet': 128,
    'UTime': 256,
    'DeepSleepNet': 128,
    'TinySleepNet': 128,
    'ResNet': 256
}



class MainModel(nn.Module):
    
    def __init__(self, config):

        super(MainModel, self).__init__()

        self.cfg = config
        self.bb_cfg = config['backbone']
        self.training_mode = config['training_params']['step']
        
        if self.bb_cfg['name'] == 'Ours':
            self.feature = Ours(self.cfg)
        elif self.bb_cfg['name'] == 'SleePyCo':
            self.feature = SleePyCoBackbone(self.cfg)
        elif self.bb_cfg['name'] == 'XSleepNet':
            self.feature = XSleepNetFeature(self.cfg)
        elif self.bb_cfg['name'] == 'UTime':
            self.feature = UTimeEncoder(self.cfg)
        elif self.bb_cfg['name'] == 'IITNet':
            self.feature = IITNetBackbone(self.cfg)
        elif self.bb_cfg['name'] == 'DeepSleepNet':
            self.feature = DeepSleepNetFeature(self.cfg)
        elif self.bb_cfg['name'] == 'TinySleepNet':
            self.feature = TinySleepNetFeature(self.cfg)
        elif self.bb_cfg['name'] == 'ResNet':
            self.feature = FeatureCNN(self.cfg)
        else:
            raise NotImplementedError('backbone not supported: {}'.format(config['backbone']['name']))

        if self.bb_cfg['dropout']:
            self.dropout = nn.Dropout(p=0.5)

        proj_dim = self.cfg['proj_head']['dim']
        if config['proj_head']['name'] == 'Linear':
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),

                nn.Flatten())
            self.head = nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim)



        elif config['proj_head']['name'] == 'MLP':
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(last_chn_dict[config['backbone']['name']], proj_dim),
                nn.ReLU(inplace=True),
                nn.Linear(proj_dim, proj_dim)
            )
            
        else:
            raise NotImplementedError('head not supported: {}'.format(config['proj_head']['name']))
        

        self.adaptive_poll = nn.AdaptiveAvgPool1d(10)

        print('[INFO] Number of params of backbone: ', sum(p.numel() for p in self.feature.parameters() if p.requires_grad))
        # print('[INFO] Number of params of proj_head: ', config['proj_head']['name'], sum(p.numel() for p in self.head.parameters() if p.requires_grad))

        if self.training_mode == 'train':
            self.classifier = get_classifier(config)
            print('[INFO] Number of params of classifier: ', sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))

    def get_max_len(self, features):
        len_list = []
        for feature in features:
            len_list.append(feature.shape[1])
        
        return max(len_list)

    def forward(self, x):

        outputs = []
        c3,c4,c5 = self.feature(x)

        if self.training_mode == 'pretrain':
            c3 = self.pool(c3)
            c4 = self.pool(c4)
            c5 = self.pool(c5)
            c3 = self.head(c3)
            c4 = self.head(c4)
            c5 = self.head(c5)
            

            return c3,c4,c5


        elif self.training_mode == 'train':
            for feature in [c3, c4,c5]:
                feature = feature.transpose(1, 2)
                feature = self.classifier(feature) 
                outputs.append(feature)    # (B, L, H)
            return outputs
        

