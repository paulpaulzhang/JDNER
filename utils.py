# coding:utf-8

import os
import random
import numpy as np
import torch
from collections import defaultdict

from torch.cuda.amp import autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.emb_name = args.emb_name
        self.epsilon = args.epsilon

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class FreeLB():
    def __init__(self, args, model, optimizer, inputs, scaler=None):
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.inputs = inputs
        self.scaler = scaler

    def attack(self):
        # ============================ Code for adversarial training=============
        # initialize delta
        if isinstance(self.model.module, torch.nn.DataParallel):
            embeds_init = self.model.module.module.encoder.embeddings.word_embeddings(
                self.inputs['input_ids'])
        else:
            embeds_init = self.model.module.encoder.embeddings.word_embeddings(
                self.inputs['input_ids'])

        if self.args.adv_init_mag > 0:

            input_mask = self.inputs['attention_mask'].to(embeds_init)
            input_lengths = torch.sum(input_mask, 1)
            # check the shape of the mask here..

            if self.args.norm_type == "l2":
                delta = torch.zeros_like(
                    embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = input_lengths * embeds_init.size(-1)
                mag = self.args.adv_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.norm_type == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.adv_init_mag,
                                                               self.args.adv_init_mag) * input_mask.unsqueeze(2)

        else:
            delta = torch.zeros_like(embeds_init)

        # the main loop
        for astep in range(self.args.adv_steps):
            # (0) forward
            delta.requires_grad_()
            self.inputs['inputs_embeds'] = delta + embeds_init

            # forward
            outputs = self.model.module(**self.inputs)

            # 计算损失
            _, adv_loss = self.model._get_train_loss(self.inputs, outputs)

            adv_loss = adv_loss / self.args.adv_steps

            adv_loss.backward()

            if astep == self.args.adv_steps - 1:
                # further updates on delta
                break

            # (2) get gradient on delta
            delta_grad = delta.grad.clone().detach()

            # (3) update and clip
            if self.args.norm_type == "l2":
                denorm = torch.norm(delta_grad.view(
                    delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr *
                         delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta_norm = torch.norm(delta.view(
                        delta.size(0), -1).float(), p=2, dim=1).detach()
                    exceed_mask = (delta_norm > self.args.adv_max_norm).to(
                        embeds_init)
                    reweights = (self.args.adv_max_norm / delta_norm * exceed_mask
                                 + (1 - exceed_mask)).view(-1, 1, 1)
                    delta = (delta * reweights).detach()
            elif self.args.norm_type == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(
                    0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr *
                         delta_grad / denorm).detach()
                if self.args.adv_max_norm > 0:
                    delta = torch.clamp(
                        delta, -self.args.adv_max_norm, self.args.adv_max_norm).detach()
            else:
                print("Norm type {} not specified.".format(self.args.norm_type))
                exit()

            if isinstance(self.model.module, torch.nn.DataParallel):
                embeds_init = self.model.module.module.encoder.embeddings.word_embeddings(
                    self.batch_cuda['input_ids'])
            else:
                embeds_init = self.model.module.encoder.embeddings.word_embeddings(
                    self.batch_cuda['input_ids'])


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_evaluate_fpr(y_true, y_pred):
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.data.cpu().numpy()
    pred = []
    true = []
    for b, l, start, end in zip(*np.where(y_pred > 0)):
        pred.append((b, l, start, end))
    for b, l, start, end in zip(*np.where(y_true > 0)):
        true.append((b, l, start, end))

    R = set(pred)
    T = set(true)
    X = len(R & T)  # X = tp
    Y = len(R)  # fp = Y - X
    Z = len(T)  # fn = Z - X
    return X, Y - X, Z - X


def compute_kl_loss(p, q, pad_mask=None):

    p_loss = F.kl_div(F.sigmoid(p),
                      F.sigmoid(q), reduction='none')
    q_loss = F.kl_div(F.sigmoid(q),
                      F.sigmoid(p), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class Logs:
    def __init__(self, path) -> None:
        self.path = path
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write('')

    def write(self, content):
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(content)
