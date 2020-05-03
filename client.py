#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from models.Nets import CNNMNIST
import json
import time
import matplotlib.pyplot as plt
import socketserver


def visualize_prediction(dataset, model):
    data_all = torch.zeros((0, 32, 32))
    plt_n_row, plt_n_col = 16, 16
    indices = torch.randint(len(dataset), (plt_n_row * plt_n_col,))

    correct = 0
    for i in indices:
        data, label = dataset[i]
        score = model.forward(data.unsqueeze(0))[0]
        pred = score.max(1).indices
        if pred == label:
            data_all = torch.cat((data_all, data))
            correct += 1
        else:
            data_all = torch.cat((data_all, -data + data.max() + data.min()))
    #        print("score = ", score, pred, pred==label)

    data_all = data_all.view((plt_n_row, plt_n_col, 32, 32))
    data_all = data_all.transpose(1, 2)
    data_all = data_all.contiguous().view((plt_n_row * 32, plt_n_col * 32))
    print(correct, plt_n_row * plt_n_col)
    plt.imshow(data_all.numpy())
    plt.draw()
    plt.pause(1)


def state_dict_to_xferable(state_dict, lr=None, ep=None):
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


def xferable_to_state_dict(xfer_string, dtype_dict, weight_shape_dict):
    import json
    import collections
    json_decode_dict = json.loads(xfer_string)
    recovered_dict = collections.OrderedDict()
    for k in json_decode_dict:
        if k != 'lr' and k != 'ep':
            recovered_dict[k] = torch.Tensor(json_decode_dict[k])
    return recovered_dict, json_decode_dict['lr'], json_decode_dict['ep']


def get_dataset_from_name(dataset_name, iid, num_users):
    # load dataset and split users
    if dataset_name == 'mnist':

        dataset_train = torch.load('./datasets/mnist/processed/training.pt')
        dataset_test = torch.load('./datasets/mnist/processed/test.pt')

    else:
        raise ValueError("Invalid dataset name")

    return dataset_train, dataset_test


def get_model_from_args(args):
    # build model
    if args.dataset == 'mnist':
        if args.model == 'CNN2':
            net_glob = CNNMNIST(args)
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
        # image, label = self.dataset[self.idxs[item]]
        image, label = self.dataset[0][self.idxs[item]], self.dataset[1][self.idxs[item]]
        m = nn.ConstantPad2d(2, 0)
        if isinstance(image, torch.Tensor):
            if image.ndimension() not in {2, 3}:
                raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(image.ndimension()))

            elif image.ndimension() == 2:
                # if 2D image, add channel dimension (CHW)
                pic = image.unsqueeze(0)
        if isinstance(pic, torch.Tensor):
            npimg = np.transpose(pic.numpy(), (1, 2, 0))
        npimg = npimg[:, :, 0]
        pic = Image.fromarray(npimg, mode='L')
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        elif pic.mode == 'F':
            img = torch.from_numpy(np.array(pic, np.float32, copy=False))
        elif pic.mode == '1':
            img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        img = img.view(pic.size[1], pic.size[0], 1)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            image = img.float().div(255)
        else:
            image = img
        image = m(image)
        image = image.clone()
        dtype = image.dtype
        mean = torch.Tensor((0.1307,))
        std = torch.Tensor((0.3081,))
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image, label


class Dict_to_namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class LocalUpdate(object):
    def __init__(self, args):
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
        dataset, _ = get_dataset_from_name(dataset_name, iid=iid, num_users=num_users)
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=bs, shuffle=True)

    def train(self, param_portable):

        # Load the newest parameters
        model = self.model
        dtype_dict = self.dtype_dict
        shape_dict = self.shape_dict
        param, lr, ep = xferable_to_state_dict(param_portable, dtype_dict, shape_dict)
        model.load_state_dict(param)

        """Train for one epoch on the training set"""
        criterion = nn.CrossEntropyLoss()
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
            zero_mat = torch.zeros(shape)
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            kldiv = nn.KLDivLoss(reduce=True)
            cost = gamma * kldiv(logsoftmax(out), softmax(zero_mat))
            dis_loss.append(cost.data.item())
            return cost

        def cal_uniform_out(out):
            shape = out.size()
            zero_mat = torch.zeros(shape)
            softmax = nn.Softmax(dim=1)
            logsoftmax = nn.LogSoftmax(dim=1)
            kldiv = nn.KLDivLoss(reduce=True)
            cost = gamma * kldiv(logsoftmax(out), softmax(zero_mat))
            dis_loss.append(cost.data.item())
            return cost

        time_register = time.time()
        for iter in range(self.args.local_ep):
            batch_loss = []
            last_w_turn = []
            for i, (input, target) in enumerate(self.ldr_train):
                if i % 50 == 0:
                    print("%4d/%4d" % (i, len(self.ldr_train)), ": %.3f" % (time.time() - time_register))

                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output, act = model(input_var)
                loss = criterion(output, target_var) * alpha

                # store activation

                # extra cost
                if self.args.loss_type == 'fedprox':
                    reg_loss = 0
                    for name, param in model.named_parameters():
                        reg_loss += l2_norm(param, w_glob[name])
                    loss += self.args.mu / 2 * reg_loss
                elif self.args.loss_type == 'uniform':
                    loss += cal_uniform_act(act)

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

        new_model = state_dict_to_xferable(model.state_dict())
        print("Training is done")

        return new_model


class MyTCPHandler(socketserver.BaseRequestHandler):

    def handle(self):
        MSGLEN = 15000000
        MSGLEN = 160000

        # self.request is the TCP socket connected to the client
        chunks = []
        bytes_recd = 0
        while bytes_recd < MSGLEN:
            chunk = self.request.recv(min(MSGLEN - bytes_recd, 8096))
            if bytes_recd == 0:
                time_model_recv_start = time.time()

            if chunk == b'':
                raise RuntimeError("socket connection broken")
            chunks.append(chunk)
            bytes_recd = bytes_recd + len(chunk)

        time_model_xmit = time.time() - time_model_recv_start
        print("Model transmission time = %.3f(s)" % time_model_xmit)

        # Decode received bytes to formatted string
        data = b''.join(chunks)
        self.data = data.decode("utf-8")

        print("{} wrote:".format(self.client_address[0]))

        # Perform a local epoch
        # w_str, loss, dis_l, sim_l, act, label = FL_node.train(self.data)
        if args_visualize_prediction:
            visualize_prediction(FL_node.ldr_train.dataset, FL_node.model)
        w_str = FL_node.train(self.data)
        # Encode to utf-8 and send back new model
        w_bytes = bytes(w_str + "\n", "utf-8")
        w_bytes += bytes(" ", "utf-8") * (MSGLEN - len(w_bytes))
        self.request.sendall(w_bytes)


def get_idxs_from_json(filename, usr_index):
    import json
    with open(filename) as f:
        dict_json_users = json.loads(f.readline())
    idxs = np.array(dict_json_users[str(usr_index)])
    return idxs


def main():
    import argparse
    from utils.options import args_parser

    args = args_parser()
    if args.usr_index is None:
        raise ValueError("args.usr_index is required")

    # Create the FL node and make it global, it will be called for training
    global FL_node
    idxs = get_idxs_from_json('dict_users.json', args.usr_index)

    global args_visualize_prediction
    args_visualize_prediction = args.visualize_prediction
    if args_visualize_prediction:
        plt.ion()
        plt.show()
    FL_node = LocalUpdate(args)
    FL_node.set_dataset(args.dataset, idxs)
    FL_node.set_model()

    # Create the server, binding to localhost on port 9999
    if args.port is None:
        raise ValueError("Need to specify port for compute node")
    HOST, PORT = args.host, args.port
    server = socketserver.TCPServer((HOST, PORT), MyTCPHandler)

    # Activate the server; this will keep running until you
    # interrupt the program with Ctrl-C
    server.serve_forever()


if __name__ == '__main__':
    main()
