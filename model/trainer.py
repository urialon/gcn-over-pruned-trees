"""
A trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from model.gcn import GCNClassifier
from utils import constant, torch_utils

class Trainer(object):
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']

    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")


def unpack_batch(batch, cuda):
    if cuda:
        inputs = [Variable(b.cuda()) for b in batch[:10]]
        labels = Variable(batch[10].cuda())
    else:
        inputs = [Variable(b) for b in batch[:10]]
        labels = Variable(batch[10])
    tokens = batch[0]
    head = batch[5]
    subj_pos = batch[6]
    obj_pos = batch[7]
    lens = batch[1].eq(0).long().sum(1).squeeze()
    return inputs, labels, tokens, head, subj_pos, obj_pos, lens

class GCNTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = GCNClassifier(opt, emb_matrix=emb_matrix)
        if opt['label_weight'] > 1:
            freq = [0.8089953613998003, 0.03586107685984381, 0.027743526510480888, 0.02237097058305443, 0.011860724561094474, 0.006869825612119077, 0.006532205977335447, 0.005607421760319417, 0.005489988843872938, 0.005724854676765897, 0.004858786917973108, 0.004770712230638248, 0.004345017908519758, 0.0041982267629616585, 0.0037872115553989785, 0.0033615172332804883, 0.0030972931712759085, 0.002627561505489989, 0.0015266279138042393, 0.002495449474487699, 0.002422053901708649, 0.002187188068815689, 0.002231225412483119, 0.0019670013504785393, 0.0017908519758088192, 0.0018202102049204392, 0.0016293817156949092, 0.0017174564030297693, 0.0015413070283600493, 0.0013357994245787094, 0.0011890082790206094, 0.0011156127062415596, 0.0011009335916857496, 0.0009247842170160296, 0.0009541424461276496, 0.0010568962480183195, 0.0007192766132346897, 0.0007779930714579297, 0.0005578063531207798, 0.0004110152075626798, 0.00033761963478362984, 8.807468733485996e-05]
            weights = [1/f for f in freq]
            scaled_weights = [w * 42 / sum(weights) for w in weights]
            self.criterion = nn.CrossEntropyLoss(weight=torch.cat([torch.ones(1), opt['label_weight']*torch.ones(opt['num_class'] - 1)], dim=-1))
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])

    def update(self, batch):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])

        # step forward
        self.model.train()
        self.optimizer.zero_grad()
        logits, pooling_output = self.model(inputs)
        
        loss = self.criterion(logits, labels)
        # l2 decay on all conv layers
        if self.opt.get('conv_l2', 0) > 0:
            loss += self.model.conv_l2() * self.opt['conv_l2']
        # l2 penalty on output representations
        if self.opt.get('pooling_l2', 0) > 0:
            loss += self.opt['pooling_l2'] * (pooling_output ** 2).sum(1).mean()
        loss_val = loss.item()
        # backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])
        self.optimizer.step()
        return loss_val

    def predict(self, batch, unsort=True):
        inputs, labels, tokens, head, subj_pos, obj_pos, lens = unpack_batch(batch, self.opt['cuda'])
        orig_idx = batch[11]
        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, labels)
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        if unsort:
            _, predictions, probs = [list(t) for t in zip(*sorted(zip(orig_idx,\
                    predictions, probs)))]
        return predictions, probs, loss.item()
