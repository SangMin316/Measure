import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transform import *



class SCL(torch.nn.modules.loss._Loss): # w/o entropy loss 
    def __init__(self,config,device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.h = config['training_params']['tau']
        self.seq_len = config['dataset']['seq_len']



    def Supcon(self,x, labels,d_labels,model):
        # x.shape = batch*seq_len, feature_size
        # x = x.view(-1,x.shape[-1])
        # x = x.unsqueeze(1)
        z = model(x)
        z = F.normalize(z, dim=1)


        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.h)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )

        exp_logits = torch.exp(logits)*logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos
        loss = loss.view(2, int(z.shape[0]/2)).mean()
        # loss = loss.mean()
        return loss
    
    
    def forward(self,x,labels,d_labels,model):
        labels = labels.view(-1,1)
        loss = self.Supcon(x,labels,d_labels,model)
        return loss






class Ours_WOH(torch.nn.modules.loss._Loss): # w/o hierarchy
    def __init__(self,config, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.beta = config['training_params']['beta']
        self.seq_len = config['dataset']['seq_len']
        bsz = config['training_params']['batch_size']
        self.D_bsz = int(bsz/2)
        self.tau = config['training_params']['tau']


    

    def Entropy(self,x,bend):
        ### Entropy
        logits = torch.div(torch.matmul(z, z.T),bend)
        # print(bend)
        # print(logits)
        logits = torch.exp(logits)
        diagonal_matrix = torch.diag(logits.sum(1)).to(self.device)
        H_devh = torch.div(torch.matmul(diagonal_matrix,z),bend)
        # H_devh = torch.matmul(diagonal_matrix,z)

        I = torch.eye(logits.size(0), device=self.device)
        K_plus_etaI  =  logits + self.eta * I
        # print(logits)
        inverse_matrix = torch.pinverse(K_plus_etaI)
        G_stein = -torch.matmul(inverse_matrix, H_devh)
        # print(G_stein)
        entropy = -(G_stein.detach()*z).sum(-1).mean()
        # print('entropy: ', entropy)
        # return   -positive 
        return  entropy
    

    def entropy_gradeint(self,z):
        dlog_q,tau = self.com_score(z)
        grad_en=torch.mean(torch.sum(-dlog_q.detach()*z,-1))
        return grad_en, tau


    @torch.no_grad()
    def com_score(self, keys,eta=0.01):
        batch_size= keys.size()[0]
        pairwise_similar=torch.mm(keys,torch.t(keys))
        # tau=self.heuristic_kernel_width(keys,keys,pairwise_similar)
        tau = torch.median(1-pairwise_similar)
        # tau = torch.abs(torch.median(pairwise_similar))
        # tau = 0.1
        Gram=torch.exp(pairwise_similar/tau)
        x_row=torch.unsqueeze(keys,-2)
        diff =x_row/tau
        grad_x =torch.sum(Gram.unsqueeze(-1)*diff,-2)
        Gram_ivs=torch.pinverse(Gram+eta*torch.eye(batch_size,device=self.device))
        
        # I = torch.eye(A.size(0), dtype=torch.float32)
        # Gram_ivs, _ = torch.solve(I, Gram+eta*torch.eye(batch_size,device=self.device))
        dlog_q= -torch.einsum('ik,kj->ij',[Gram_ivs,grad_x])
        return  dlog_q, tau



    def Supcon(self,z, labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)
        mask = mask.repeat(2, 2)
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )



        negative_mask = 1 - mask
        negative_mask = negative_mask*logits_mask

        exp_logits = torch.exp(logits)*negative_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean() 
        # loss = loss.mean()
        return loss

    def forward(self,x,labels,d_labels,model):
        _,_,z = model(x)
        # z = model(x)

        z = F.normalize(z, dim=1)


        labels = labels.view(-1,1)
        d_labels0 = d_labels.unsqueeze(1).repeat(1, self.seq_len).view(-1)

        scl_loss =  self.Supcon(z,labels,d_labels)


        # print(d_labels0)
        entropy = 0
        avg_z = 0
        if x.shape[0] == int(self.D_bsz*4):
            z_1 = torch.cat((z[0:self.D_bsz,:], z[2*self.D_bsz:3*self.D_bsz,:]), dim=0)
            z_2 = torch.cat((z[self.D_bsz:2*self.D_bsz,:], z[3*self.D_bsz:4*self.D_bsz,:]), dim=0)
            # ent1 = self.Entropy(x1,model,bend)
            # print('ent1: ', ent1)
            # ent2 = self.Entropy(x2,model,bend)
            ent1, avg_z1 = self.entropy_gradeint(z_1)
            ent2, avg_z2 = self.entropy_gradeint(z_2)
            entropy += (ent1 + ent2)/2
            avg_z += (avg_z1 + avg_z2)/2

        else:
            ent1, avg_z1 = self.entropy_gradeint(z)
            avg_z += avg_z1
            entropy += ent1
        
        loss = -self.beta*entropy + scl_loss

        return loss, 1-avg_z, entropy





class SCL_HM(torch.nn.modules.loss._Loss):
    def __init__(self,config, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.seq_len = config['dataset']['seq_len']
        bsz = config['training_params']['batch_size']
        self.D_bsz = int(bsz/2)
        self.tau = config['training_params']['tau']


    def Supcon(self,z, labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)
        mask = mask.repeat(2, 2)
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )


        exp_logits = torch.exp(logits)*logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean() 
        # loss = loss.mean()
        return loss

    def forward(self,x,labels,d_labels,model):
        _,z2,z3 = model(x)
        z2 = F.normalize(z2, dim=1)
        z3 = F.normalize(z3, dim=1)
        labels = labels.view(-1,1)
        d_labels0 = d_labels.unsqueeze(1).repeat(1, self.seq_len).view(-1)

        scl_loss2 =  self.Supcon(z2,labels,d_labels)
        scl_loss3 =  self.Supcon(z3,labels,d_labels)

        scl_loss =  (scl_loss2+ scl_loss3)/2

        
        loss = scl_loss 

        return loss






class Ours_HML(torch.nn.modules.loss._Loss): # 
    def __init__(self,config, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.beta = config['training_params']['beta']
        self.seq_len = config['dataset']['seq_len']
        bsz = config['training_params']['batch_size']
        self.D_bsz = int(bsz/2)
        self.tau = config['training_params']['tau']

    def Align(self,z,labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float()
        mask = mask.repeat(2, 2)
        
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)
        d_mask = 1- d_mask

        
        # compute logitsf
        logits =  torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability

        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()  # remove diagonal elements
        # mask = mask*d_mask
        # logits = torch.matmul(z, z.T)

        positive = -logits*mask
        positive = positive.sum(1) / (mask.sum(1) + 1e-9)
        positive = positive.mean()
        return  positive
    

    def Entropy(self,x,bend):
        ### Entropy
        logits = torch.div(torch.matmul(z, z.T),bend)
        # print(bend)
        # print(logits)
        logits = torch.exp(logits)
        diagonal_matrix = torch.diag(logits.sum(1)).to(self.device)
        H_devh = torch.div(torch.matmul(diagonal_matrix,z),bend)
        # H_devh = torch.matmul(diagonal_matrix,z)

        I = torch.eye(logits.size(0), device=self.device)
        K_plus_etaI  =  logits + self.eta * I
        # print(logits)
        inverse_matrix = torch.pinverse(K_plus_etaI)



        G_stein = -torch.matmul(inverse_matrix, H_devh)
        # print(G_stein)
        entropy = -(G_stein.detach()*z).sum(-1).mean()
        # print('entropy: ', entropy)
        # return   -positive 
        return  entropy
    

    def entropy_gradeint(self,z):
        dlog_q,tau = self.com_score(z)
        grad_en=torch.mean(torch.sum(-dlog_q.detach()*z,-1))
        return grad_en, tau


    @torch.no_grad()
    def com_score(self, keys,eta=0.01):
        batch_size= keys.size()[0]
        pairwise_similar=torch.mm(keys,torch.t(keys))
        # tau=self.heuristic_kernel_width(keys,keys,pairwise_similar)
        tau = torch.median(1-pairwise_similar)
        # tau = torch.abs(torch.median(pairwise_similar))
        # tau = 0.1
        Gram=torch.exp(pairwise_similar/tau)
        x_row=torch.unsqueeze(keys,-2)
        diff =x_row/tau
        grad_x =torch.sum(Gram.unsqueeze(-1)*diff,-2)
        Gram_ivs=torch.pinverse(Gram+eta*torch.eye(batch_size,device=self.device))
        
        # I = torch.eye(A.size(0), dtype=torch.float32)
        # Gram_ivs, _ = torch.solve(I, Gram+eta*torch.eye(batch_size,device=self.device))
        dlog_q= -torch.einsum('ik,kj->ij',[Gram_ivs,grad_x])
        return  dlog_q, tau

    def Supcon(self,z, labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)
        mask = mask.repeat(2, 2)
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )

        # negative_mask = (1-mask) * (1 - d_mask) # other domain = 1
        # negative_mask = 1 - negative_mask
        # negative_mask = negative_mask*logits_mask


        negative_mask = 1 - mask
        negative_mask = negative_mask*logits_mask


        exp_logits = torch.exp(logits)*negative_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos.mean() 
        # loss = loss.mean()
        return loss

    def forward(self,x,labels,d_labels,model):
        z1,z2,z3 = model(x)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        z3 = F.normalize(z3, dim=1)
        labels = labels.view(-1,1)
        d_labels0 = d_labels.unsqueeze(1).repeat(1, self.seq_len).view(-1)

        scl_loss1 =  self.Supcon(z1,labels,d_labels)
        scl_loss2 =  self.Supcon(z2,labels,d_labels)
        scl_loss3 =  self.Supcon(z3,labels,d_labels)


        scl_loss = (scl_loss1 + scl_loss2+ scl_loss3)/3

        # print(d_labels0)
        entropy = 0
        avg_z = 0
        for z in [z1,z2,z3]:
            # if x.shape[0] == int(self.D_bsz*4):
            if len(torch.unique(d_labels)) == 2:
                z_1 = torch.cat((z[0:self.D_bsz,:], z[2*self.D_bsz:3*self.D_bsz,:]), dim=0)
                z_2 = torch.cat((z[self.D_bsz:2*self.D_bsz,:], z[3*self.D_bsz:4*self.D_bsz,:]), dim=0)
                # ent1 = self.Entropy(x1,model,bend)
                # print('ent1: ', ent1)
                # ent2 = self.Entropy(x2,model,bend)
                ent1, avg_z1 = self.entropy_gradeint(z_1)
                ent2, avg_z2 = self.entropy_gradeint(z_2)
                entropy += (ent1 + ent2)/2
                avg_z += (avg_z1 + avg_z2)/2

            else:
                ent1, avg_z1 = self.entropy_gradeint(z)
                avg_z += avg_z1
                entropy += ent1
        entropy = entropy/3
        avg_z = avg_z/3
        
        loss = -self.beta*entropy + scl_loss 
        # loss = -self.beta*entropy + scl_loss + align_loss

        return loss, 1-avg_z, entropy




class Ours_HL(torch.nn.modules.loss._Loss): #High-level and Low level
    def __init__(self,config, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.beta = config['training_params']['beta']
        self.seq_len = config['dataset']['seq_len']
        bsz = config['training_params']['batch_size']
        self.D_bsz = int(bsz/2)
        self.tau = config['training_params']['tau']

    def Align(self,z,labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float()
        mask = mask.repeat(2, 2)
        
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)
        d_mask = 1- d_mask

        
        # compute logitsf
        logits =  torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability


        positive = -logits*mask
        positive = positive.sum(1) / (mask.sum(1) + 1e-9)
        positive = positive.mean()
        return  positive
    

    def Entropy(self,x,bend):
        ### Entropy
        logits = torch.div(torch.matmul(z, z.T),bend)
        # print(bend)
        # print(logits)
        logits = torch.exp(logits)
        diagonal_matrix = torch.diag(logits.sum(1)).to(self.device)
        H_devh = torch.div(torch.matmul(diagonal_matrix,z),bend)
        # H_devh = torch.matmul(diagonal_matrix,z)

        I = torch.eye(logits.size(0), device=self.device)
        K_plus_etaI  =  logits + self.eta * I
        # print(logits)
        inverse_matrix = torch.pinverse(K_plus_etaI)



        G_stein = -torch.matmul(inverse_matrix, H_devh)
        # print(G_stein)
        entropy = -(G_stein.detach()*z).sum(-1).mean()
        # print('entropy: ', entropy)
        # return   -positive 
        return  entropy
    

    def entropy_gradeint(self,z):
        dlog_q,tau = self.com_score(z)
        grad_en=torch.mean(torch.sum(-dlog_q.detach()*z,-1))
        return grad_en, tau


    @torch.no_grad()
    def com_score(self, keys,eta=0.01):
        batch_size= keys.size()[0]
        pairwise_similar=torch.mm(keys,torch.t(keys))
        # tau=self.heuristic_kernel_width(keys,keys,pairwise_similar)
        tau = torch.median(1-pairwise_similar)
        # tau = torch.abs(torch.median(pairwise_similar))
        # tau = 0.1
        Gram=torch.exp(pairwise_similar/tau)
        x_row=torch.unsqueeze(keys,-2)
        diff =x_row/tau
        grad_x =torch.sum(Gram.unsqueeze(-1)*diff,-2)
        Gram_ivs=torch.pinverse(Gram+eta*torch.eye(batch_size,device=self.device))
        
        # I = torch.eye(A.size(0), dtype=torch.float32)
        # Gram_ivs, _ = torch.solve(I, Gram+eta*torch.eye(batch_size,device=self.device))
        dlog_q= -torch.einsum('ik,kj->ij',[Gram_ivs,grad_x])
        return  dlog_q, tau

    def Supcon(self,z, labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)
        mask = mask.repeat(2, 2)
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )


        negative_mask = 1 - mask
        negative_mask = negative_mask*logits_mask

        exp_logits = torch.exp(logits)*negative_mask

        # exp_logits = torch.exp(logits)*logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        # loss = -(mask_outd * logits).sum(1) / (mask_outd.sum(1) + 1e-9) - (mask_ind * logits).sum(1) / (mask_ind.sum(1) + 1e-9) + torch.log(exp_logits.sum(1, keepdim=True))
        loss = -mean_log_prob_pos.mean() 
        # loss = loss.mean()
        return loss

    def forward(self,x,labels,d_labels,model):
        z1,z2,z3 = model(x)
        z1 = F.normalize(z1, dim=1)
        # z2 = F.normalize(z2, dim=1)
        z3 = F.normalize(z3, dim=1)
        labels = labels.view(-1,1)
        d_labels0 = d_labels.unsqueeze(1).repeat(1, self.seq_len).view(-1)

        scl_loss1 =  self.Supcon(z1,labels,d_labels)
        # scl_loss2 =  self.Supcon(z2,labels,d_labels)
        scl_loss3 =  self.Supcon(z3,labels,d_labels)



        scl_loss = (scl_loss1 + scl_loss3)/2

        # print(d_labels0)
        entropy = 0
        avg_z = 0
        for z in [z1,z3]:
            # if x.shape[0] == int(self.D_bsz*4):
            if len(torch.unique(d_labels)) == 2:
                z_1 = torch.cat((z[0:self.D_bsz,:], z[2*self.D_bsz:3*self.D_bsz,:]), dim=0)
                z_2 = torch.cat((z[self.D_bsz:2*self.D_bsz,:], z[3*self.D_bsz:4*self.D_bsz,:]), dim=0)
                # ent1 = self.Entropy(x1,model,bend)
                # print('ent1: ', ent1)
                # ent2 = self.Entropy(x2,model,bend)
                ent1, avg_z1 = self.entropy_gradeint(z_1)
                ent2, avg_z2 = self.entropy_gradeint(z_2)
                entropy += (ent1 + ent2)/2
                avg_z += (avg_z1 + avg_z2)/2

            else:
                ent1, avg_z1 = self.entropy_gradeint(z)
                avg_z += avg_z1
                entropy += ent1
        entropy = entropy/2
        avg_z = avg_z/2
        
        loss = -self.beta*entropy + scl_loss 
        # loss = -self.beta*entropy + scl_loss + align_loss

        return loss, 1-avg_z, entropy




class Ours_HL(torch.nn.modules.loss._Loss): # High-level and Middle level
    def __init__(self,config, device):
        super().__init__()
        self.device = device
        self.criterion = torch.nn.CrossEntropyLoss().to(device)
        self.beta = config['training_params']['beta']
        self.seq_len = config['dataset']['seq_len']
        bsz = config['training_params']['batch_size']
        self.D_bsz = int(bsz/2)
        self.tau = config['training_params']['tau']

    def Align(self,z,labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float()
        mask = mask.repeat(2, 2)
        
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)
        d_mask = 1- d_mask

        
        # compute logitsf
        logits =  torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        # logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        # logits = logits - logits_max.detach()  # remove diagonal elements
        # mask = mask*d_mask
        # logits = torch.matmul(z, z.T)

        positive = -logits*mask
        positive = positive.sum(1) / (mask.sum(1) + 1e-9)
        positive = positive.mean()
        return  positive
    

    def Entropy(self,x,bend):
        ### Entropy
        logits = torch.div(torch.matmul(z, z.T),bend)
        # print(bend)
        # print(logits)
        logits = torch.exp(logits)
        diagonal_matrix = torch.diag(logits.sum(1)).to(self.device)
        H_devh = torch.div(torch.matmul(diagonal_matrix,z),bend)

        I = torch.eye(logits.size(0), device=self.device)
        K_plus_etaI  =  logits + self.eta * I
        inverse_matrix = torch.pinverse(K_plus_etaI)



        G_stein = -torch.matmul(inverse_matrix, H_devh)
        entropy = -(G_stein.detach()*z).sum(-1).mean()
        return  entropy
    

    def entropy_gradeint(self,z):
        dlog_q,tau = self.com_score(z)
        grad_en=torch.mean(torch.sum(-dlog_q.detach()*z,-1))
        return grad_en, tau


    @torch.no_grad()
    def com_score(self, keys,eta=0.01):
        batch_size= keys.size()[0]
        pairwise_similar=torch.mm(keys,torch.t(keys))
        # tau=self.heuristic_kernel_width(keys,keys,pairwise_similar)
        tau = torch.median(1-pairwise_similar)
        # tau = torch.abs(torch.median(pairwise_similar))
        # tau = 0.1
        Gram=torch.exp(pairwise_similar/tau)
        x_row=torch.unsqueeze(keys,-2)
        diff =x_row/tau
        grad_x =torch.sum(Gram.unsqueeze(-1)*diff,-2)
        Gram_ivs=torch.pinverse(Gram+eta*torch.eye(batch_size,device=self.device))
        
        # I = torch.eye(A.size(0), dtype=torch.float32)
        # Gram_ivs, _ = torch.solve(I, Gram+eta*torch.eye(batch_size,device=self.device))
        dlog_q= -torch.einsum('ik,kj->ij',[Gram_ivs,grad_x])
        return  dlog_q, tau

    def Supcon(self,z, labels,d_labels):
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(self.device)
        mask = mask.float().to(self.device)
        mask = mask.repeat(2, 2)
        d_labels = d_labels.contiguous().view(-1, 1)
        d_mask = torch.eq(d_labels, d_labels.T).float().to(self.device)
        d_mask = d_mask.repeat(2, 2)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(z, z.T),
            self.tau)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # remove diagonal elements

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(z.shape[0]).view(-1, 1).to(self.device),
            0
        )



        negative_mask = 1 - mask
        negative_mask = negative_mask*logits_mask
   
        exp_logits = torch.exp(logits)*negative_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        loss = -mean_log_prob_pos.mean() 
        return loss

    def forward(self,x,labels,d_labels,model):
        z1,z2,z3 = model(x)
        z2 = F.normalize(z2, dim=1)
        z3 = F.normalize(z3, dim=1)
        labels = labels.view(-1,1)
        d_labels0 = d_labels.unsqueeze(1).repeat(1, self.seq_len).view(-1)

        scl_loss2 =  self.Supcon(z2,labels,d_labels)
        scl_loss3 =  self.Supcon(z3,labels,d_labels)

   
        scl_loss = ( scl_loss2+ scl_loss3)/2

        entropy = 0
        avg_z = 0
        for z in [z2,z3]:
            if len(torch.unique(d_labels)) == 2:
                z_1 = torch.cat((z[0:self.D_bsz,:], z[2*self.D_bsz:3*self.D_bsz,:]), dim=0)
                z_2 = torch.cat((z[self.D_bsz:2*self.D_bsz,:], z[3*self.D_bsz:4*self.D_bsz,:]), dim=0)

                ent1, avg_z1 = self.entropy_gradeint(z_1)
                ent2, avg_z2 = self.entropy_gradeint(z_2)
                entropy += (ent1 + ent2)/2
                avg_z += (avg_z1 + avg_z2)/2

            else:
                ent1, avg_z1 = self.entropy_gradeint(z)
                avg_z += avg_z1
                entropy += ent1
        entropy = entropy/2
        avg_z = avg_z/2
        
        loss = -self.beta*entropy + scl_loss 

        return loss, 1-avg_z, entropy





