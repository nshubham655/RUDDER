"""This module contains an implementation of the max margin ranking loss, slightly
modified from this code:
https://github.com/antoine77340/Mixture-of-Embedding-Experts/blob/master/loss.py

The modification is the `fix_norm` conditional, which removes zero terms from the
diagonal when performing the averaging calculation.

Original licence below.
"""

# Copyright 2018 Antoine Miech All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
import torch as th
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import ot

class PartialOrderLoss(nn.Module):
    def __init__(self,margin=0.05,margin2=0.09,fix_norm=True, caption_count=28, gamma=1.0, p= 0.05):
        super().__init__()
        self.margin1 = margin
        self.margin2 = margin2
        self.caption_count = caption_count
        self.p = p
      
    
    def forward(self, dist, labels):
        # print('------------------------------------------------------')
        # print('margins', self.margin1, self.margin2)
        # print('dist shape', dist.shape, dist)
        # print(labels)
        zeros_helper = torch.zeros(dist.shape).cuda()
        scores=dist.cuda()
        #print('substracted score shape', scores[:,0].view(scores.size(0),1).shape)
        scores = scores-scores[:,0].view(scores.size(0),1).expand_as(scores)
        #print('scores shape', scores.shape, scores)
        part = torch.where(labels==0, (-self.p-scores).clamp(min=0) + scores.clamp(min=0), zeros_helper).clamp(min=0)
        part1 = torch.where((labels==1) | (labels==2), (self.margin1+scores).clamp(min=0) + (-scores-self.margin2).clamp(min=0), zeros_helper).clamp(min=0)
        part2 = torch.where(labels==3, (self.margin2+scores), zeros_helper).clamp(min=0)
        #print('part1, part2',part1.shape, part2.shape, part1, part2)
        scores = dist[:,self.caption_count:].cuda()
        diagonal = scores.diag().view(scores.size(0), 1)
        d = diagonal.t().expand_as(scores)
        scores = scores - d
        #print('scores shape', scores.shape, scores)
        part3 = (self.margin2+scores).clamp(min=0)
        #print('part3 shape', part3.shape, part3)
        mask = torch.eye(scores.size(0)).cuda() > 0.5
        part3 = part3.masked_fill_(mask,0)
        #print('part3 shape', part3.shape, part3)
        Loss = part1.mean()+part2.mean()+part.mean()+part3.mean()
        return Loss
    
class OTPartialOrderLoss(nn.Module):
    def __init__(self,margin=0.05,margin2=0.09,fix_norm=True, caption_count=21, gamma=1.0, p= 0.02):
        super().__init__()
        self.margin1 = margin
        self.margin2 = margin2
        self.caption_count = caption_count
        self.p = p
        self.gamma = gamma
    #def compute_loss(self, scores):
        
    
    def forward(self, dist, labels):
        zeros_helper = torch.zeros(dist.shape).cuda()
        scores=dist.cuda()
        scores = scores-scores[:,0].view(scores.size(0),1).expand_as(scores)
        groundMetric0 = torch.nn.functional.relu(-self.p - scores)+torch.nn.functional.relu(scores)
        groundMetric1 = torch.nn.functional.relu(self.margin1 + scores)
        groundMetric2 = torch.nn.functional.relu(-self.margin2 - scores)
        groundMetric3 = torch.nn.functional.relu(self.margin2 + scores)
        GM0 = (labels==0).cuda().float().mul(groundMetric0)
        GM1 = ((labels==1) | (labels==2)).cuda().float().mul(groundMetric1)
        GM2 = (((labels==1) | (labels==2))).cuda().float().mul(groundMetric2)# why not 1 or 2 result better without not    
        GM3 = (labels==3).cuda().float().mul(groundMetric3)
        GM=GM0+GM1+GM2+GM3
        # GM=(~(labels==0)).cuda().float().mul(GM)#why prune label zero losss
        expGM = torch.exp(-self.gamma * GM)
        GMFlatten = expGM.view(-1)

        uuu = np.ones([dist.size(0)]) / dist.size(0)
        vvv = np.ones([dist.size(1)]) / dist.size(1)
        reg = (-1) / 5
        expGM_numpy = expGM.cpu().detach().numpy()
        
        T = torch.from_numpy(ot.sinkhorn(uuu, vvv, expGM_numpy, reg, numItermax=50)).cuda().float()
        # T = (~(labels==0)).cuda().float().mul(T) why remove label 0
        Tsum = T.sum(dim=1).view(-1,1)
        T = T/Tsum
        T_Flatten = torch.autograd.Variable(T.view(-1)).float().cuda()

        loss1 = GM.view(-1).mul(T_Flatten).mean()
        

        scores = dist[:,self.caption_count:].cuda()
        diagonal = scores.diag().view(scores.size(0), 1)
        d = diagonal.t().expand_as(scores)
        scores = scores - d
        
        
        
        
        target_train = torch.eye(scores.size(0)).cuda()
        
        hinge_groundMetric = torch.nn.functional.relu(self.margin2 + scores)
        Pos_groundMetric = torch.nn.functional.relu(-0.5-scores)
        GM_PositivePair = target_train.mul(Pos_groundMetric)

        GM_NegativePair = (1 - target_train).mul(hinge_groundMetric)
        GM = GM_PositivePair + GM_NegativePair
        GM = GM.masked_fill_(target_train>0.5,0)
        GMF = GM.view(-1)
        
        expGM = torch.exp(-self.gamma * GM)

        GMFlatten = expGM.view(-1)

        uuu = np.ones([scores.size(0)]) / scores.size(0)
        vvv = np.ones([scores.size(1)]) / scores.size(1)
        reg = (-1) / 5
        expGM_numpy = expGM.cpu().detach().numpy()
        # print('I am busy at computing sinkhorn')
        # sys.stdout.flush()
        
        scores = target_train.mul((-0.5-scores).clamp(min=0)) + (1-target_train).mul((self.margin2+scores).clamp(min=0))
        T = torch.from_numpy(ot.sinkhorn(uuu, vvv, expGM_numpy, reg, numItermax=50)).cuda()
        T = T.masked_fill_(target_train>0.5, 0)
        Tsum = T.sum(dim=1).view(-1,1)
        T = T/Tsum       
        T_Flatten = torch.autograd.Variable(T.view(-1)).float().cuda()
        loss2 = scores.view(-1).mul(T_Flatten).mean()
        
        Loss = loss1+loss2
        return Loss
            
        


class OptimalTransportLoss(nn.Module):
    def __init__(self,margin=0.05,margin2=0.09,fix_norm=True, caption_count=28, gamma=1.0, p= 0.05):
        super().__init__()
        self.margin = margin
        self.caption_count = caption_count
        self.gamma = gamma
    def optimal_transport(self, dist,labels):
        dist=dist.cuda()
        # print(dist.size(0))
        # print(dist.size(1))
        if dist.size(0) < dist.size(1):
            captions_per_video = int(dist.size(1)/dist.size(0))
            diagonal_mask = torch.eye(dist.size(0)).repeat(1,captions_per_video).cuda()
            reweight_mask = diagonal_mask.clone()
            reweight_mask[:,:dist.size(0)]=1
        else:
            captions_per_video = int(dist.size(0)/dist.size(1))
            diagonal_mask = torch.eye(dist.size(1)).repeat(captions_per_video,1).cuda()

        diagonal = diagonal_mask*dist
        diagonal.masked_fill_(diagonal_mask < 0.5, -1e28)
        max_diagonal, max_inds = diagonal.max(dim=1)
        d1 = max_diagonal.view(-1,1).expand_as(dist)


        target_train = diagonal_mask
        maxelement_mask = torch.zeros(dist.size()).cuda()
        maxelement_mask[torch.arange(dist.size(0)),max_inds] = 1
        bsz = dist.size(0)

        #target = torch.from_numpy(np.arange(0,bsz)).float().cuda()
        #target = target.view(target.size(0), 1)
        #target_train = (target == torch.transpose(target, 0, 1)).float().cuda()


        hinge_groundMetric = torch.nn.functional.relu(self.margin + dist - d1)
        Pos_groundMetric = torch.nn.functional.relu(-0.5-dist+d1)
        GM_PositivePair = target_train.mul(Pos_groundMetric)

        GM_NegativePair = (1 - target_train).mul(hinge_groundMetric)
        GM = GM_PositivePair + GM_NegativePair
        GM = GM.masked_fill_(target_train>0.5,0)
        GMF = GM.view(-1)
        
        expGM = torch.exp(-self.gamma * GM)

        GMFlatten = expGM.view(-1)

        uuu = np.ones([dist.size(0)]) / dist.size(0)
        vvv = np.ones([dist.size(1)]) / dist.size(1)
        reg = (-1) / 5
        expGM_numpy = expGM.cpu().detach().numpy()
        # print('I am busy at computing sinkhorn')
        # sys.stdout.flush()
        
        dist = target_train.mul((-0.5-dist+d1).clamp(min=0)) + (1-target_train).mul((self.margin+dist-d1).clamp(min=0))
        T = torch.from_numpy(ot.sinkhorn(uuu, vvv, expGM_numpy, reg, numItermax=50)).cuda()
        T = T.masked_fill_(maxelement_mask>0.5, 0)
        Tsum = T.sum(dim=1).view(-1,1)
        T = T/Tsum
        if dist.size(0) < dist.size(1):
            T = T.masked_fill_(reweight_mask<0.5, 0)
            Tsum = T.sum(dim=1).view(-1,1)
            T = T/Tsum#dist = dist.masked_fill_(I, 0)
        T_Flatten = torch.autograd.Variable(T.view(-1)).float().cuda()
        
        loss = torch.sum(dist.view(-1).mul(T_Flatten))

        return loss
    def forward(self, dist, labels):
        dist=dist[:,self.caption_count:]
        return self.optimal_transport(dist,labels)+self.optimal_transport(dist.t(),labels)


class MaxMarginRankingLoss(nn.Module):
    def __init__(self,margin=0.05,margin2=0.09,fix_norm=True, caption_count=28, gamma=1.0, p= 0.05):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = th.nn.MarginRankingLoss(margin)
        self.margin = margin
        self.caption_count = caption_count
    def forward(self, x, labels):
        x = x[:,self.caption_count:]
        #print(x.shape)
        #exit()
        n = x.size()[0]

        x1 = th.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = th.cat((x1, x1), 0)

        x2 = x.contiguous().view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)
        x2 = th.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = th.ones(x.shape) - th.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = th.nonzero(th.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = th.index_select(x1, dim=0, index=keep_idx)
            x2_ = th.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

class DistanceWeightedLoss(nn.Module):

    def __init__(self,margin=0.09,margin2=0.09,fix_norm=True, caption_count=21, gamma=1.0, p= 0.02):
        super().__init__()
        self.margin = 0.09
        self.nonzero_loss_cutoff = self.margin
        self.cutoff = -0.03
        self.caption_count = caption_count

    def forward(self, x, labels):
        gt = -x[:,0]+1
        x = x[:,self.caption_count:]
        n = x.size()[0]
        d = x.size()[1]

        zeros_helper = torch.zeros(x.shape).cuda()
        ones_helper = torch.ones(x.shape).cuda()
        scores = -x.cuda()+1
        distance = scores.cuda()
        scores_temp = torch.where(distance - gt.view(scores.size(0),1).expand_as(scores) < self.cutoff, zeros_helper+1e-10, distance)
        log_weights = ((2.0 - float(d)) * torch.log(scores_temp) - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (scores_temp ** 2.0)))
        weights = torch.exp(log_weights - torch.max(log_weights))
        mask = (1-torch.eye(n)).cuda()
        weights = weights * mask * torch.where(distance - gt.view(scores.size(0),1).expand_as(scores) > self.nonzero_loss_cutoff, ones_helper, zeros_helper+1e-10)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True)+1e-10)
        loss = torch.ones(n).cuda()
        np_weights = weights.cpu().detach().numpy()
        for i in range(n):
            try:
                idx = np.random.choice(n, 5, p=np_weights[i]).tolist()
                for j in idx:
                    loss[i] += (self.margin + distance[i][i] - distance[i][j]).clamp(min=0)
            except:
                idx = np.random.choice(n, 5).tolist()
                for j in idx:
                    loss[i] += (self.margin + distance[i][i] - distance[i][j]).clamp(min=0)
        
        distance = scores.t().cuda()
        scores_temp = torch.where(distance - gt.view(scores.size(0),1).expand_as(scores) < self.cutoff, zeros_helper+1e-10, distance)
        log_weights = ((2.0 - float(d)) * torch.log(scores_temp) - (float(d - 3) / 2) * torch.log(1.0 - 0.25 * (scores_temp ** 2.0)))
        weights = torch.exp(log_weights - torch.max(log_weights))
        mask = (1-torch.eye(n)).cuda()
        weights = weights * mask * torch.where(distance - gt.view(scores.size(0),1).expand_as(scores) > self.nonzero_loss_cutoff, ones_helper, zeros_helper+1e-10)
        weights = weights / (torch.sum(weights, dim=1, keepdim=True)+1e-10)
        loss1 = torch.ones(n).cuda()
        np_weights = weights.cpu().detach().numpy()
        for i in range(n):
            try:
                idx = np.random.choice(n, 10, p=np_weights[i]).tolist()
                for j in idx:
                    loss1[i] += (self.margin + distance[i][i] - distance[i][j]).clamp(min=0)
            except:
                idx = np.random.choice(n, 10).tolist()
                for j in idx:
                    loss1[i] += (self.margin + distance[i][i] - distance[i][j]).clamp(min=0)
        
        return loss.mean()+loss1.mean()


class QuadrupletLoss(nn.Module):
    def __init__(self,margin=0.05,margin2=0.09,fix_norm=True, caption_count=28, gamma=1.0, p= 0.05):
        super().__init__()
        self.margin1 = margin
        self.margin2 = margin2
        self.caption_count = caption_count
      
    
    def forward(self, dist, labels):
        zeros_helper = torch.zeros(dist.shape).cuda()
        scores=dist.cuda()
        # for p+ and p-
        # Dap-
        temp = torch.max(torch.where((labels==1) or (labels==2), scores, zeros_helper)) - scores[:,0]
        # Dap+
        score_temp = scores[:, 0] = 2
        temp1 = torch.min(torch.where((labels==0), score_temp[:, :self.caption_count], zeros_helper[:, :self.caption_count]))
        temp2 = torch.where(temp1==0, 1, temp1) - scores[:,0]
        loss1 = (self.margin2-self.margin1+temp2-temp)/(temp2+self.margin2-self.margin1)
        loss1 = loss1.clamp(min=0)
        
        # for p- and n
        # Dan
        temp3 = torch.max(torch.where((labels==3), scores, zeros_helper)) - scores[:,0]        
        # Dap-
        temp4 = torch.min(torch.where((labels==1) or (labels==2), scores, zeros_helper))
        temp5 = torch.where(temp4==0, 1, temp4) - scores[:,0]
        loss2 = (self.margin1+temp5-temp3)/(temp5+self.margin1)
        loss2 = loss2.clamp(min=0)

        return loss1.mean()+loss2.mean()


class BCEWithLogitsLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = th.nn.BCEWithLogitsLoss(weight=weight)

    def forward(self, x, target):
        return self.loss(x, target)


class CrossEntropyLoss(nn.Module):

    def __init__(self, weight=None):
        super().__init__()
        self.loss = th.nn.CrossEntropyLoss(weight=weight)

    def forward(self, x, target):
        return self.loss(x, target.long().to(x.device))


if __name__ == "__main__":
    loss = BCEWithLogitsLoss()
    x = th.randn(3, requires_grad=True)
    target = th.empty(3).random_(2)
    output = loss(x, target)
    output.backward()
    print(target)
