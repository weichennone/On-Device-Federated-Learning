#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser('On board federated learning')
    parser.add_argument('--dataset', default='mnist', type=str)
    parser.add_argument('--weight-decay', type=float, default=1e-04)
    # parser.add_argument('--dropout-prob', type=float, default=0)
    parser.add_argument('--lr', type=float, default=.1)
    # parser.add_argument('--lr-decay', type=float, default=.1)
    parser.add_argument('--lr_drop', type=float, default=.992)
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
    parser.add_argument('--epochs', type=int, default=200)
    # parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--test-size', type=int, default=100)

    # federated arguments
    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=1, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=100, help="local batch size: B")
    parser.add_argument('--resume', type=str, default='trained.tar')

    # other arguments
    parser.add_argument('--model', type=str, default='CNNStd', help="name of model")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # loss function related
    # parser.add_argument('--loss_type', type=str, default='none')

    #   uniform distribution
    # parser.add_argument('--alpha', type=float, default=1.0)
    # parser.add_argument('--beta', type=float, default=0.0)
    # parser.add_argument('--gamma', type=float, default=0.0)

    #   FedProx
    # parser.add_argument('--mu', type=float, default=0.0)

    # Implementation
    parser.add_argument('--usr_index', type=int, default=None)
    parser.add_argument('--remote_index', type=int, nargs='+')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int)
    parser.add_argument('--visualize_prediction', action='store_true')

    args = parser.parse_args()
    return args
