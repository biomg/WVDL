#-- coding:utf8 --
import sys
import math
import os
import numpy as np
np.set_printoptions(threshold=np.inf)
import pdb

import torch    
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from collections import OrderedDict

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
import random
import gzip
import pickle
import timeit
import argparse

if torch.cuda.is_available():
        cuda = True
        #torch.cuda.set_device(1)
        print('===> Using GPU')
else:
        cuda = False
        print('===> Using CPU')


def padding_sequence_new(seq, window_size = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < window_size:
        gap_len = window_size -seq_len
        new_seq = seq + repkey * gap_len
    return new_seq


def padding_sequence(seq, max_len = 501, repkey = 'N'):
    seq_len = len(seq)
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq


def get_RNA_seq_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()

    return new_array


def split_overlap_seq(seq, window_size):
    overlap_size = 50
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - window_size)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - window_size)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(int(num_ins)):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1, window_size)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1, window_size)
            bag_seqs.append(pad_seq)
    return bag_seqs


def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    return seq_list, labels


def get_bag_data(data, channel = 7, window_size = 101):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seqs = split_overlap_seq(seq, window_size = window_size)
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_seq_concolutional_array(bag_seq)
            bag_subt.append(tri_fea.T)
        num_of_ins = len(bag_subt)
        if num_of_ins > channel:
            start = (num_of_ins - channel)/2
            bag_subt = bag_subt[start: start + channel]
        if len(bag_subt) <channel:
            rand_more = channel - len(bag_subt)
            for ind in range(rand_more):
                tri_fea = get_RNA_seq_concolutional_array('N'*window_size)
                bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels


def get_bag_data_1_channel(data, max_len = 501):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    for seq in seqs:
        bag_seq = padding_sequence(seq, max_len = max_len)
        bag_subt = []
        tri_fea = get_RNA_seq_concolutional_array(bag_seq)
        bag_subt.append(tri_fea.T)
        bags.append(np.array(bag_subt))
    return bags, labels


def read_data_file(posifile, negafile = None, train = True):
    data = dict()
    seqs, labels = read_seq_graphprot(posifile, label = 1)
    if negafile:
        seqs2, labels2 = read_seq_graphprot(negafile, label = 0)
        seqs = seqs + seqs2
        labels = labels + labels2
        # print(labels)
        
    data["seq"] = seqs
    data["Y"] = np.array(labels)
    
    return data

def get_data(posi, nega = None, channel = 7,  window_size = 101, train = True):
    data = read_data_file(posi, nega, train = train)
    if channel == 1:
        train_bags, label = get_bag_data_1_channel(data, max_len = window_size)

    else:
        train_bags, label = get_bag_data(data, channel = channel, window_size = window_size)
    
    return train_bags, label



class Estimator(object):

    def __init__(self, model):
        self.model = model

    def compile(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_f = loss

    def _fit(self, train_loader):
        """
        train one epoch
        """
        loss_list = []
        for idx, (X, y) in enumerate(train_loader):
            X_v = Variable(X)
            y_v = Variable(y)
            # print np.array(X_v).shape
            if cuda:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
            self.optimizer.zero_grad()
            y_pred = self.model(X_v)
            loss = self.loss_f(y_pred, y_v)
            loss.backward()
            self.optimizer.step()

            ## for log
            loss_list.append(loss.item()) # need change to loss_list.append(loss.item()) for pytorch v0.4 or above

        return sum(loss_list) / len(loss_list)

    def fit(self, X, y, batch_size=32, nb_epoch=10, validation_data=()):
        print (X.shape)
        train_set = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)).long().view(-1))
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        self.model.train()
       
        
        for t in range(nb_epoch):
            loss = self._fit(train_loader)
            # train_loss_print.append(loss)
            print("Epoch %s/%s loss: %06.4f" % (t, nb_epoch, loss))
       


    def evaluate(self, X, y, batch_size=32):
        
        y_pred = self.predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if cuda:
            y_v = y_v.cuda()
        loss = self.loss_f(y_pred, y_v)
        predict = y_pred.data.cpu().numpy()[:, 1].flatten()
        auc = roc_auc_score(y, predict)
        #lasses = torch.topk(y_pred, 1)[1].data.numpy().flatten()
        #cc = self._accuracy(classes, y)
        return loss.item(), auc

    def _accuracy(self, y_pred, y):
        return float(sum(y_pred == y)) / y.shape[0]

    def predict(self, X):
        X = Variable(torch.from_numpy(X.astype(np.float32)))
        if cuda:
            X= X.cuda()        
        y_pred = self.model(X)
        return y_pred        

    def predict_proba(self, X):
        self.model.eval()
        return self.model.predict_proba(X)


class CNN(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        out1_size = int((window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool_size = int((out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(nb_filter, nb_filter, kernel_size = (1, 10), stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU(),
            nn.MaxPool2d(pool_size, stride = stride))
        out2_size = int((maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1)
        maxpool2_size = int((out2_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1)
        self.drop1 = nn.Dropout(p=0.25)
        # print ('maxpool_size', maxpool_size)
        self.fc1 = nn.Linear(maxpool2_size*nb_filter, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
    
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]
    
class CNN_LSTM(nn.Module):
    def __init__(self, nb_filter, channel = 7, num_classes = 2, kernel_size = (4, 10), pool_size = (1, 3), labcounts = 32, window_size = 12, hidden_size = 200, stride = (1, 1), padding = 0, num_layers = 2):
        super(CNN_LSTM, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(channel, nb_filter, kernel_size, stride = stride, padding = padding),
            nn.BatchNorm2d(nb_filter),
            nn.ReLU())
        self.pool1 = nn.MaxPool2d(pool_size, stride = stride)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        out1_size = (window_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1] + 1
        maxpool_size = (out1_size + 2*padding - (pool_size[1] - 1) - 1)/stride[1] + 1
        self.downsample = nn.Conv2d(nb_filter, 1, kernel_size = (1, 10), stride = stride, padding = padding)
        input_size = int((maxpool_size + 2*padding - (kernel_size[1] - 1) - 1)/stride[1]) + 1
        self.layer2 = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True, bidirectional=True)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(2*hidden_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.pool1(out)
        # print(out.shape)
        out = self.downsample(out)
        # print(out.shape)
        out = torch.squeeze(out, 1)
        # print(out.shape)
        #pdb.set_trace()
        if cuda:
            x = x.cuda()
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda() 
            # print(h0.shape)
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)).cuda()
        else:
            h0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size)) 
            c0 = Variable(torch.zeros(self.num_layers*2, out.size(0), self.hidden_size))
        out, _  = self.layer2(out, (h0, c0))
        # print(out.shape)
        out = out[:, -1, :]
        #pdb.set_trace()
        # print(out.shape)
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        # x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.layer1(x)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]

def convR(in_channels, out_channels, kernel_size, stride=1, padding = (0, 1)):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, stride=stride, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channel, nb_filter = 16, kernel_size = (1, 3), stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convR(in_channel, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn1 = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convR(nb_filter, nb_filter, kernel_size = kernel_size, stride = stride)
        self.bn2 = nn.BatchNorm2d(nb_filter)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers, nb_filter = 16, channel = 7, labcounts = 12, window_size = 36, kernel_size = (1, 3), pool_size = (1, 3), num_classes=2, hidden_size = 200):
        super(ResNet, self).__init__()
        self.in_channels = channel
        self.conv = convR(self.in_channels, nb_filter, kernel_size = (4, 10))
        cnn1_size = window_size - 7
        self.bn = nn.BatchNorm2d(nb_filter)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, nb_filter, layers[0],  kernel_size = kernel_size)
        self.layer2 = self.make_layer(block, nb_filter*2, layers[1], 1, kernel_size = kernel_size, in_channels = nb_filter)
        self.layer3 = self.make_layer(block, nb_filter*4, layers[2], 1, kernel_size = kernel_size, in_channels = 2*nb_filter)
        self.avg_pool = nn.AvgPool2d(pool_size)
        avgpool2_1_size = int((cnn1_size - (pool_size[1] - 1) - 1)/pool_size[1]) + 1
        last_layer_size = 4*nb_filter*avgpool2_1_size
        self.fc = nn.Linear(last_layer_size, hidden_size)
        self.drop2 = nn.Dropout(p=0.25)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1,  kernel_size = (1, 10), in_channels = 16):
        downsample = None
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                convR(in_channels, out_channels, kernel_size = kernel_size, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(in_channels, out_channels, kernel_size = kernel_size, stride = stride, downsample = downsample))
        #self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, kernel_size = kernel_size))
        return nn.Sequential(*layers)
     
    def forward(self, x):
        #print x.data.cpu().numpy().shape
        #x = x.view(x.size(0), 1, x.size(1), x.size(2))
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        #pdb.set_trace()
        #print self.layer2
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        #pdb.set_trace()
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.drop2(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        return out

    def layer1out(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        temp = out.data.cpu().numpy()
        return temp
        
    def predict_proba(self, x):
        if type(x) is np.ndarray:
            x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            x = Variable(x)
        #x = Variable(x, volatile=True)
        if cuda:
            x = x.cuda()
        y = self.forward(x)
        temp = y.data.cpu().numpy()
        return temp[:, 1]






def train_network(model_type, X_train, y_train, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16, motif = False, motif_seqs = [], motif_outdir = 'motifs'):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CNN, CNN-LSTM, ResNet model')

    if cuda:
        model = model.cuda()
    clf = Estimator(model)
    clf.compile(optimizer=torch.optim.Adam(model.parameters(), lr=0.001, weight_decay = 0.0001),
                loss=nn.CrossEntropyLoss())
    clf.fit(X_train, y_train, batch_size=batch_size, nb_epoch=n_epochs)

    torch.save(model.state_dict(), model_file)
    #print 'predicting'         
    #pred = model.predict_proba(test_bags)
    #return model

def predict_network(model_type, X_test, channel = 7, window_size = 107, model_file = 'model.pkl', batch_size = 100, n_epochs = 50, num_filters = 16):
    print ('model training for ', model_type)
    #nb_epos= 5
    if model_type == 'CNN':
        model = CNN(nb_filter =num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'CNNLSTM':
        model = CNN_LSTM(nb_filter = num_filters, labcounts = 4, window_size = window_size, channel = channel)
    elif model_type == 'ResNet':
        model = ResNet(ResidualBlock, [3, 3, 3], nb_filter = num_filters, labcounts = 4, channel = channel , window_size = window_size)
    else:
        print ('only support CNN, CNN-LSTM, ResNet model')

    if cuda:
        model = model.cuda()
                
    model.load_state_dict(torch.load(model_file))
    try:
        pred = model.predict_proba(X_test)
    except: #to handle the out-of-memory when testing
        test_batch = batch(X_test)
        pred = []
        for test in test_batch:
            pred_test1 = model.predict_proba(test)[:, 1]
            pred = np.concatenate((pred, pred_test1), axis = 0)
    return pred
        

def batch(tensor, batch_size = 1000):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

def run(parser):
    posi = parser.posi
    nega = parser.nega
    out_file = parser.out_file
    train = parser.train
    model_file = parser.model_file
    predict = parser.predict
    batch_size = parser.batch_size
    n_epochs = parser.n_epochs
    num_filters = parser.num_filters
    testfile = parser.testfile
    start_time = timeit.default_timer()
    
    #pdb.set_trace() 
    if predict:
        train = False
        if testfile == '':
            print ('you need specify the fasta file for predicting when predict is True')
            return
    if train:
        if posi == '' or nega == '':
            print ('you need specify the training positive and negative fasta file for training when train is True')
            return

    if train:
        file_out = open('time_train.txt','a')
        # print("ResNet-transformer-4")
        motif_seqs = []
        data = read_data_file(posi, nega)
        motif_seqs = data['seq']

        print("101")
        train_bags, train_labels = get_data(posi, nega, channel = 7, window_size = 101)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 7, window_size = 101 + 6, model_file = model_type + '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        
        print("151")
        train_bags, train_labels = get_data(posi, nega, channel = 4, window_size = 151)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 4, window_size = 151 + 6, model_file = model_type + '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        print("201")
        train_bags, train_labels = get_data(posi, nega, channel = 3, window_size = 201)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 3, window_size = 201 + 6, model_file = model_type + '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        print("251")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 251)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 251 + 6, model_file = model_type + '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)


        print("301")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 301)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 301 + 6, model_file = model_type + '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        

        print("351")
        train_bags, train_labels = get_data(posi, nega, channel = 2, window_size = 351)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 2, window_size = 351 + 6, model_file = model_type + '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        

        print("401")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 401)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 401 + 6, model_file = model_type + '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        

        print("451")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 451)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 451 + 6, model_file = model_type + '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        

        print ("501")
        train_bags, train_labels = get_data(posi, nega, channel = 1, window_size = 501)
        model_type = "CNN"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "CNNLSTM"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        model_type = "ResNet"
        train_network(model_type, np.array(train_bags), np.array(train_labels), channel = 1, window_size = 501 + 6, model_file = model_type + '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        

        end_time = timeit.default_timer()
        file_out.write(str(round(float(end_time - start_time),3))+'\n')
        file_out.close()
        # print ("Training final took: %.2f min" % float((end_time - start_time)/60))
    elif predict:
        fw = open(out_file, 'w')
        file_out = open('pre_auc.txt','a')
        file_out2 = open('time_test.txt', 'a')

        
        model_type = "CNN"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        CnnPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        CnnPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        CnnPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        CnnPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        CnnPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        model_type = "CNNLSTM"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        CnnLstmPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        CnnLstmPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        CnnLstmPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        CnnLstmPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        CnnLstmPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        CnnLstmPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        CnnLstmPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        CnnLstmPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        CnnLstmPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        model_type = "ResNet"
        X_test, X_labels = get_data(testfile, nega , channel = 7, window_size = 101)
        ResNetPre1 = predict_network(model_type, np.array(X_test), channel = 7, window_size = 101 + 6, model_file = model_type+ '.101', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 4, window_size = 151)
        ResNetPre2 = predict_network(model_type, np.array(X_test), channel = 4, window_size = 151 + 6, model_file = model_type+ '.151', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 3, window_size = 201)
        ResNetPre3 = predict_network(model_type, np.array(X_test), channel = 3, window_size = 201 + 6, model_file = model_type+ '.201', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 251)
        ResNetPre4 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 251 + 6, model_file = model_type+ '.251', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 301)
        ResNetPre5 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 301 + 6, model_file = model_type+ '.301', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 2, window_size = 351)
        ResNetPre6 = predict_network(model_type, np.array(X_test), channel = 2, window_size = 351 + 6, model_file = model_type+ '.351', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 401)
        ResNetPre7 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 401 + 6, model_file = model_type+ '.401', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 451)
        ResNetPre8 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 451 + 6, model_file = model_type+ '.451', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)
        X_test, X_labels = get_data(testfile, nega , channel = 1, window_size = 501)
        ResNetPre9 = predict_network(model_type, np.array(X_test), channel = 1, window_size = 501 + 6, model_file = model_type+ '.501', batch_size = batch_size, n_epochs = n_epochs, num_filters = num_filters)

        CnnPre = (CnnPre1 + CnnPre2 + CnnPre3 + CnnPre4 + CnnPre5 + CnnPre6 + CnnPre7 + CnnPre8 + CnnPre9) / 9
        CnnLstmPre = (CnnLstmPre1 + CnnLstmPre2 + CnnLstmPre3 + CnnLstmPre4 + CnnLstmPre5 + CnnLstmPre6 + CnnLstmPre7 + CnnLstmPre8 + CnnLstmPre9) / 9
        ResNetPre = (ResNetPre1 + ResNetPre2 + ResNetPre3 + ResNetPre4 + ResNetPre5 + ResNetPre6 + ResNetPre7 + ResNetPre8 + ResNetPre9) / 9

        for i in range(1,9):
            for j in range(1,10-i):
                predict = (i*CnnPre + j*CnnLstmPre + (10-i-j)*ResNetPre)/30
                auc = roc_auc_score(X_labels, predict)
                print ('AUC:{:.3f}'.format(auc))        
                myprob = "\n".join(map(str, predict))  
                fw.write(myprob)
                file_out.write(str(round(float(auc),3))+'\n')

        
        fw.close()
        file_out.close()
        end_time = timeit.default_timer()
        file_out2.write(str(round(float(end_time - start_time),3))+'\n')
        file_out2.close()
    else:
        print ('please specify that you want to train the mdoel or predict for your own sequences')


def parse_arguments(parser):
    parser.add_argument('--posi', type=str, metavar='<postive_sequecne_file>', help='The fasta file of positive training samples')
    parser.add_argument('--nega', type=str, metavar='<negative_sequecne_file>', help='The fasta file of negative training samples')
    parser.add_argument('--out_file', type=str, default='prediction.txt', help='The output file used to store the prediction probability of the testing sequences')
    parser.add_argument('--train', type=bool, default=False, help='The path to the Pickled file containing durations between visits of patients. If you are not using duration information, do not use this option')
    parser.add_argument('--model_file', type=str, default='model.pkl', help='The file to save model parameters. Use this option if you want to train on your sequences or predict for your sequences')
    parser.add_argument('--predict', type=bool, default=False,  help='Predicting the RNA-protein binding sites for your input sequences, if using train, then it will be False')
    parser.add_argument('--testfile', type=str, default='',  help='the test fast file for sequences you want to predict for, you need specify it when using predict')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch (default value: 100)')
    parser.add_argument('--num_filters', type=int, default=16, help='The number of filters for CNNs (default value: 16)')
    parser.add_argument('--n_epochs', type=int, default=50, help='The number of training epochs (default value: 50)')
    args = parser.parse_args()    #解析添加的参数
    return args

parser = argparse.ArgumentParser()
args = parse_arguments(parser)
print (args)
run(args)