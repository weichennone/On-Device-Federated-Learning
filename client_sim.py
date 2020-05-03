#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy
import numpy as np
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid_1, cifar_noniid_2, emnist_noniid_1, femnist_star, cifar_100_noniid, cifar_100_iid
from models.Nets import CNNMNIST


def state_dict_to_xferable( state_dict, lr=None, ep=None ):
    import json
    xfer_dict = {}
    if lr is not None:
        xfer_dict['lr'] = lr
    if ep is not None:
        xfer_dict['ep'] = ep
    for k in state_dict:
        xfer_dict[k] = state_dict[k].cpu().tolist()
    xfer_string = json.dumps(xfer_dict)
    return xfer_string


def xferable_to_state_dict( xfer_string, dtype_dict, weight_shape_dict ):
    import json
    import collections
    json_decode_dict = json.loads(xfer_string)
    recovered_dict = collections.OrderedDict()
    for k in json_decode_dict:
        if k != 'lr' and k != 'ep':
            recovered_dict[k] = torch.Tensor(json_decode_dict[k])
    return recovered_dict, json_decode_dict['lr'], json_decode_dict['ep']


def get_dataset_from_name( dataset_name, iid, num_users ):
    # load dataset and split users
    if dataset_name == 'cifar10':
        _CIFAR_TRAIN_TRANSFORMS = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_train = datasets.CIFAR10(
            './datasets/cifar10', train=True, download=True,
            transform=transforms.Compose(_CIFAR_TRAIN_TRANSFORMS))

        _CIFAR_TEST_TRANSFORMS = [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)
            ),
        ]
        dataset_test = datasets.CIFAR10(
            './datasets/cifar10', train=False,
            transform=transforms.Compose(_CIFAR_TEST_TRANSFORMS))

        if iid == 0:
            dict_users = cifar_iid(dataset_train, num_users)
        elif iid == 1:
            dict_users = cifar_noniid_1(dataset_train, num_users)
        elif iid == 2:
            dict_users = cifar_noniid_2(dataset_train, num_users)
        else:
            exit('Error: unrecognized class')
    elif dataset_name == 'mnist':
        _MNIST_TRAIN_TRANSFORM = _MNIST_TEST_TRANSFORM = [
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Pad(2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        dataset_train = datasets.MNIST(
            './datasets/mnist', train=True, download=True,
            transform=transforms.Compose(_MNIST_TRAIN_TRANSFORM)
        )
        dataset_test = datasets.MNIST(
            './datasets/mnist', train=False, download=True,
            transform=transforms.Compose(_MNIST_TEST_TRANSFORM)
        )
        if iid == 0:
            dict_users = mnist_iid(dataset_train, num_users)
        elif iid == 2:
            dict_users = mnist_noniid(dataset_train, num_users)
    else:
        raise ValueError("Invalid dataset name")

    return dataset_train, dataset_test


def get_model_from_args( args ):
    # build model
    if args.dataset == 'mnist':
        if args.model == 'CNN2':
            net_glob = CNNMNIST(args).cuda()
    else:
        exit('Error: unrecognized model')
    return net_glob


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class Dict_to_namespace(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

class LocalUpdate(object):
    def __init__(self, args_dict):
        if type(args_dict) is not dict:
            raise ValueError()
        args = Dict_to_namespace(args_dict)
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []

        self.dtype_dict = {}
        self.shape_dict = {}

    def set_model(self):
        model = get_model_from_args(self.args)
        self.model = model
        for name, param in model.named_parameters():
            self.dtype_dict[name] = param.detach().cpu().numpy().dtype
            self.shape_dict[name] = param.detach().cpu().numpy().shape

    def set_dataset(self, dataset_name, idxs):
        bs = self.args.batch_size
        iid = self.args.iid
        num_users = self.args.num_users
        dataset, _ = get_dataset_from_name( dataset_name, iid=iid, num_users=num_users )
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=bs, shuffle=True)

    def train(self, param_portable):

        # Load the newest parameters
        model = self.model
        dtype_dict = self.dtype_dict
        shape_dict = self.shape_dict
        param, lr, ep = xferable_to_state_dict( param_portable, dtype_dict, shape_dict )
        model.load_state_dict(param)

        """Train for one epoch on the training set"""
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    nesterov=self.args.nesterov,
                                    weight_decay=self.args.weight_decay)

        # hyper-parameters
        alpha = self.args.alpha
        beta = self.args.beta
        gamma = self.args.gamma

        # switch to train mode
        w_glob = model.state_dict()
        l2_norm = nn.MSELoss()
        model.train()

        dis_loss = []
        sim_loss = []

        def cal_uniform_act(out):
            shape = out.size()
            zero_mat = torch.zeros(shape).cuda()
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            kldiv = nn.KLDivLoss(reduce=True)
            cost = gamma * kldiv(logsoftmax(out), softmax(zero_mat))
            dis_loss.append(cost.data.item())
            return cost

        def cal_uniform_out(out):
            shape = out.size()
            zero_mat = torch.zeros(shape).cuda()
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            kldiv = nn.KLDivLoss(reduce=True)
            cost = gamma * kldiv(logsoftmax(out), softmax(zero_mat))
            dis_loss.append(cost.data.item())
            return cost

        epoch_loss = []
        last_w = '0'
        labels = torch.Tensor().cuda()
        acts = torch.Tensor().cuda()
        outputs = torch.Tensor().cuda()

        for iter in range(self.args.local_ep):
            batch_loss = []
            last_w_turn = []
            for i, (input, target) in enumerate(self.ldr_train):

                target = target.cuda()
                input = input.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output, act = model(input_var)
                loss = criterion(output, target_var) * alpha

                # store activation

                if labels.size() == torch.Size([0]):
                    labels = copy.deepcopy(target)
                else:
                    labels = torch.cat((labels, target))
                if acts.size() == torch.Size([0]):
                    acts = act.clone().detach()
                else:
                    acts = torch.cat((acts, act.clone().detach()))
                if outputs.size() == torch.Size([0]):
                    outputs = output.clone().detach()
                else:
                    outputs = torch.cat((outputs, output.clone().detach()))

                # extra cost
                if self.args.loss_type == 'fedprox':
                    reg_loss = 0
                    for name, param in model.named_parameters():
                        reg_loss += l2_norm(param, w_glob[name])
                    loss += self.args.mu / 2 * reg_loss
                elif self.args.loss_type == 'uniform':
                    loss += cal_uniform_act(act)
                    #cal_uniform_act(act)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                batch_loss.append(loss.item())

                # detect NAN
                for k in model.state_dict():
                    if torch.isnan(torch.sum(model.state_dict()[k])):
                        print(last_w)
                        print(last_w_turn)
                        exit(1)
                last_w = ' '.join(
                    ['{:.4f}'.format(float(torch.sum(model.state_dict()[l]))) for l in model.state_dict()])
                last_w_turn.append(last_w)

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        ind_select = torch.randint(len(labels), (40,))
        new_model = state_dict_to_xferable(model.state_dict())
        return new_model

