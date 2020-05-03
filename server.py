#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from utils.sampling import mnist_iid, mnist_noniid
from utils.options import args_parser
from client_sim import LocalUpdate
from models.Nets import CNNMNIST

import torch.backends.cudnn as cudnn
import json
import socket
import time


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


def xferable_to_state_dict( xfer_string ):
    import json
    import collections
    json_decode_dict = json.loads(xfer_string)
    recovered_dict = collections.OrderedDict()
    for k in json_decode_dict:
        if k != 'lr' and k != 'ep':
            recovered_dict[k] = torch.Tensor(json_decode_dict[k])
    return recovered_dict


def save_checkpoint(state, filename='trained.tar'):
    torch.save(state, filename)


def get_devices_info( filename ):
    with open(filename) as f:
        string = f.readline()
    json_obj = json.loads(string)
    return json_obj


class Server:

    def __init__(self, args):
        self.args = args
        self.lr = args.lr
        self.lr_drop = args.lr_drop
        self.devices = get_devices_info("FL_nodes.json")
        self.MSGLEN = 160000
        # load dataset and split users
        if args.dataset == 'mnist':
            _MNIST_TRAIN_TRANSFORMS = _MNIST_TEST_TRANSFORMS = [
                transforms.ToTensor(),
                transforms.ToPILImage(),
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
            self.dataset_train = datasets.MNIST(
                './datasets/mnist', train=True, download=True,
                transform=transforms.Compose(_MNIST_TRAIN_TRANSFORMS)
            )
            self.dataset_test = datasets.MNIST(
                './datasets/mnist', train=False, download=True,
                transform=transforms.Compose(_MNIST_TEST_TRANSFORMS)
            )
            if args.iid == 0:
                self.dict_users = mnist_iid(self.dataset_train, args.num_users)
            elif args.iid == 2:
                self.dict_users = mnist_noniid(self.dataset_train, args.num_users)
        else:
            exit('Error: unrecognized dataset')

        # build model
        if args.dataset == 'mnist':
            if args.model == "CNN2":
                self.net_glob = CNNMNIST(args).cuda()
            else:
                exit('Error: unrecognized model')
        else:
            exit('Error: unrecognized model')

            # Pre-allocated LocalUpdate list
            self.local_update_list = [None] * args.num_users
            args_dict = vars(args)

            dict_json_users = {}
            for d in self.dict_users:
                dict_json_users[d] = self.dict_users[d].tolist()
            s = json.dumps(dict_json_users)
            with open("dict_users.json", "w") as f:
                f.write(s)
            for i in range(args.num_users):
                self.local_update_list[i] = LocalUpdate(args_dict=args_dict)
                self.local_update_list[i].set_model()
                self.local_update_list[i].set_dataset(dataset_name=args.dataset, idxs=self.dict_users[i])

            dtype_dict = {}
            shape_dict = {}
            for name, param in self.net_glob.named_parameters():
                dtype_dict[name] = param.detach().cpu().numpy().dtype
                shape_dict[name] = param.detach().cpu().numpy().shape

        self.test_acc = []
        self.elapsed_time = []
        self.w_locals = []
        self.start_time = {}

    def _select_clients(self):
        m = max(int(self.args.frac * self.args.num_users), 1)

        self.idxs_users = np.random.choice(range(self.args.num_users),
                                           m,
                                           replace=False)
        # Socks list
        self.socks = [None] * len(self.idxs_users)

    def send_model(self):
        for i, idx in enumerate(self.idxs_users):
            print('user: {:d}'.format(idx))
            xfer_dict = state_dict_to_xferable(self.net_glob.state_dict(),
                                               lr=self.lr,
                                               ep=iter)
            print("xfer_dict = ", type(xfer_dict), len(xfer_dict))
            self.start_time[idx] = time.time()

            # Broadcast the model to all remote workers
            if idx in self.args.remote_index:
                node_name = "node%d" % idx
                HOST = self.devices[node_name]['ip']
                PORT = self.devices[node_name]['port']

                w_bytes = bytes(xfer_dict + "\n" + " " * (self.MSGLEN - len(xfer_dict) - 1), "utf-8")
                if len(w_bytes) != self.MSGLEN:
                    raise Exception("w_bytes should be equal to MSGLEN(%d)" % self.MSGLEN)

                # Create a socket (SOCK_STREAM means a TCP socket)
                self.socks[i] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socks[i].connect((HOST, PORT))
                self.socks[i].sendall(w_bytes)

            # Do the local training if no remote workers available
            else:
                w_str = self.local_update_list[idx].train(param_portable=xfer_dict)

                w = xferable_to_state_dict(w_str)

                self.w_locals.append(copy.deepcopy(w))

    def get_models(self):
        # Receive the updated model from all remote workers
        elapsed = []
        for i, idx in enumerate(self.idxs_users):
            elapsed.append(time.time() - self.start_time[idx])
            # print('user: {:d}, elapsed time: {:.4f}'.format(idx, elapsed))
            self.elapsed_time.append(max(elapsed))
            if idx in self.args.remote_index:
                node_name = "node%d" % idx
                HOST = self.devices[node_name]['ip']
                PORT = self.devices[node_name]['port']

                # Receive updated model from worker nodes
                chunks = []
                bytes_recd = 0
                while bytes_recd < self.MSGLEN:
                    chunk = self.socks[i].recv(min(self.MSGLEN - bytes_recd, 8096))
                    if chunk == b'':
                        raise RuntimeError("socket connection broken")
                    chunks.append(chunk)
                    bytes_recd = bytes_recd + len(chunk)

                # Decode the received bytes into formatted string
                data = b''.join(chunks)
                w_str = data.decode("utf-8")

                # Decode the formatted string to the model
                w = xferable_to_state_dict(w_str)
                self.w_locals.append(copy.deepcopy(w))

        # Ensure receiving all the updated weight
        if len(self.w_locals) != len(self.idxs_users):
            err_msg = "w_locals only has %d weight update (Ideally should be %d" % (len(self.w_locals), len(self.idxs_users))
            raise ValueError(err_msg)

    def fedavg(self, w):
        w_avg = copy.deepcopy(w[0])
        for k in w_avg.keys():
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))

        return w_avg

    def _adjust_learning_rate(self):
        self.lr *= self.lr_drop

    def training(self):
        self.net_glob.train()
        # w_glob = self.net_glob.state_dict()

        # Original
        for iter in range(self.args.epochs):

            self.w_locals = []
            self.start_time = {}

            self.send_model()

            self.get_models()

            # update global weights
            w_glob = self.fedavg(self.w_locals)

            # copy weight to net_glob
            self.net_glob.load_state_dict(w_glob)

            # print loss
            acc_test, _ = self.test()
            print("test accuracy: {:.4f}".format(acc_test))
            self.test_acc.append(acc_test)

            print('elapsed time: {:.4f}'.format(self.elapsed_time[-1]))

            self._adjust_learning_rate()

        print("test accuracy:")
        print(self.test_acc)

    def test(self):
        self.net_glob.eval()
        # testing
        test_loss = 0
        correct = 0
        data_loader = DataLoader(self.dataset_test, batch_size=self.args.test_size)
        l = len(data_loader)
        for idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            log_probs, _ = self.net_glob(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        test_loss /= len(data_loader.dataset)
        accuracy = 100.00 * float(correct) / len(data_loader.dataset)
        if self.args.verbose:
            print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
                test_loss, correct, len(data_loader.dataset), accuracy))
        return accuracy, test_loss


def main():
    # parse args
    args = args_parser()
    np.random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    server = Server(args)
    server.training()


if __name__ == '__main__':
    main()
