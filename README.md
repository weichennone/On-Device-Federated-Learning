# Federated Learning
This experiment is designed for verifying the hardware performance of Federated Learning algorithm on eight Raspberry Pi devices.
The experiments on MNIST dataset with two-layer CNN.


## Requirements
python 3.6

pytorch 0.4

## Usage
1. momentum: SGD momentum
2. model: the model used for training (for this experiment, use CNN2 only)
3. epochs: total communication rounds
4. seed: random seed
5. iid: 0 represents IID case, 2 represents non-IID case where each client has no more than 2 classes
6. num_users: how many nodes involved in the training (use 8 in our case)
7. frac: the percentage of whole devices that involved in the training (1 represents all devices are involved in the training)
8. dataset: MNIST
9. loss_type: none represents FedAvg
10. lr_drop: the decay of learning rate
11. local_ep: epochs of local training
12. remote_index: address which device is from remote
### To run the server
```
python main_fed.py --momentum 0.1 --model CNN2 --epochs 100 --seed 1 --iid 2 --num_users 8 --frac 1 --dataset mnist --loss_type none --lr_drop 0.996 --local_ep 5 --remote_index 0 1 2 3 4 5 6 7
```
There is a json file ```FL_nodes.json``` indicates the information (IP, port) for each node. In command line, server can choose to use remote device as node i by adding i into argument ```remote_index```
### To run the compute node
```
time python3 -m models.UpdateNode --dataset mnist --usr_index <user id> --momentum 0.1 --model CNN2 --loss_type none --lr_drop 0.996 --local_ep 5 --host 128.2.58.126 --port 9000
```
host and port should be configured to match the ```FL_nodes.json``` as above
