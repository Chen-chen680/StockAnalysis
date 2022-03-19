import torch.optim as optim
from tqdm import tqdm
from torch import nn
import torch
from model.LSTM import MyNet
from utils.dataset import TestDataset, TrainDataset, TrainDatasetOnlyShoupan, TestDatasetOnlyShoupan
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np
import time

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def TimePlus(fomattime):
    fomattime = time.strptime(fomattime, '%Y-%m-%d')
    outtime = time.mktime(fomattime) + 86400
    outtime = time.strftime('%Y-%m-%d', time.localtime(outtime))
    return outtime

def train():
    #定义超参数
    device = torch.device('cuda')
    input_size = 10
    seq_len = 30
    hidden_size = 32
    output_size = 1

    #定义模型
    net = MyNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, seq_len=seq_len)
    net.to(device, dtype=torch.float32)

    # 定义数据集
    train_dataset = TrainDataset(r'上证指数.csv', 30)
    test_dataset = TestDataset(r'上证指数.csv', 30)
    max_list, min_list = train_dataset.max_list, train_dataset.min_list

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    loss__record_list = []
    total_loss = 0.0
    for epoch in tqdm(range(3000)):
        for index, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype = torch.float32)
            label = torch.unsqueeze(label, dim=1)

            pred = net(data)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1, 'loss: ', total_loss)
            loss__record_list.append(total_loss)
            torch.save(net, 'lstm_parameters.pth')
            total_loss = 0.0
        torch.save(net, 'lstm_parameters.pth')

    with open('loss.log', 'w' ,encoding='utf8') as f:
        f.write('loss值')
        for loss in loss__record_list:
            f.write('%f\n' % loss)
    pred_list, label_list = [], []
    time_list = test_dataset.time_list
    date_list = [datetime.strptime(d, '%Y-%m-%d').date() for d in time_list]
    for index, (data, label) in enumerate(test_loader):
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)
        pred = net(data)
        pred = torch.squeeze(pred)
        pred1, label = float(pred), float(label)
        pred1 = pred1 * (max_list[1] - min_list[1]) + min_list[1]
        label = label * (max_list[1] - min_list[1]) + min_list[1]
        pred_list.append(pred1)
        label_list.append(label)
    with open('test_result.csv','w', encoding='utf-8-sig') as f:
        f.write('日期,实际值,预测值\n')
        for index in range(len(date_list)):
            f.write('%s,%f,%f\n' % (time_list[index], label_list[index], pred_list[index]))
    plt.plot(date_list, pred_list, label='预测值')
    plt.plot(date_list, label_list, label='实际值')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(date_list[::50])
    plt.gcf().autofmt_xdate()
    plt.title('股票预测')
    plt.legend()
    plt.show()

def trainOnlyShoupan():
    #定义超参数
    device = torch.device('cuda')
    input_size = 1
    seq_len = 30
    hidden_size = 32
    output_size = 1

    #定义模型
    net = MyNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size, seq_len=seq_len)
    net.to(device, dtype=torch.float32)

    # 定义数据集
    train_dataset = TrainDatasetOnlyShoupan(r'上证指数.csv', 30)
    test_dataset = TestDatasetOnlyShoupan(r'上证指数.csv', 30)
    max_value, min_value = train_dataset.max_value, train_dataset.min_value

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # 定义损失函数和优化器
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())
    loss__record_list = []
    total_loss = 0.0
    for epoch in tqdm(range(300)):
        for index, (data, label) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device, dtype=torch.float32)
            label = label.to(device, dtype = torch.float32)
            label = torch.unsqueeze(label, dim=1)
            data = torch.unsqueeze(data, dim=2)
            pred = net(data)
            loss = loss_function(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print('Epoch:', epoch + 1, 'loss: ', total_loss)
            loss__record_list.append(total_loss)
            torch.save(net, 'lstm_parameters.pth')
            total_loss = 0.0
        torch.save(net, 'lstm_parameters.pth')

    with open('loss.log', 'w' ,encoding='utf8') as f:
        f.write('loss值')
        for loss in loss__record_list:
            f.write('%f\n' % loss)
    pred_list, label_list = [], []
    time_list = test_dataset.time_list
    date_list = [datetime.strptime(d, '%Y-%m-%d').date() for d in time_list]

    for index, (data, label) in enumerate(test_loader):
        data = data.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)
        data = torch.unsqueeze(data, dim=2)
        pred = net(data)
        pred = torch.squeeze(pred)
        pred1, label = float(pred), float(label)
        pred1 = pred1 * (max_value - min_value) + min_value
        label = label * (max_value - min_value) + min_value
        pred_list.append(pred1)
        label_list.append(label)

    with open('test_result.csv','w', encoding='utf-8-sig') as f:
        f.write('日期,实际值,预测值\n')
        for index in range(len(date_list)):
            f.write('%s,%f,%f\n' % (time_list[index], label_list[index], pred_list[index]))
    plt.plot(date_list, pred_list, label='预测值')
    plt.plot(date_list, label_list, label='实际值')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(date_list[::50])
    plt.gcf().autofmt_xdate()
    plt.title('股票预测')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    trainOnlyShoupan()