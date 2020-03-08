import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
import random

from params import *

torch.manual_seed(1)


def save_restore(save_flag):
    save_root = 'pkls/'
    save_path = save_root + 'rnn_mag_net_params.pkl'
    if save_flag:
        torch.save(rnn.state_dict(), save_path)
    else:
        rnn.load_state_dict(torch.load(save_path))


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)

    def forward(self, x, h1, h2):
        timestep = x.size()[1]
        if h1 is None:
            r_out, (h1, h2) = self.lstm(x, None)
        else:
            h1 = h1.data
            h2 = h2.data
            r_out, (h1, h2) = self.lstm(x, (h1, h2))
        r_out = r_out.view(-1, HIDDEN_SIZE)
        outs = self.out(r_out)
        outs = outs.view(-1, timestep, OUTPUT_SIZE)
        return outs, (h1, h2)


def train():
    is_print = True
    # h_state = None
    h1 = None
    h2 = None
    for epoch in range(EPOCH):
        print(epoch)
        # start, end = step * np.pi, (step + 1) * np.pi
        # steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)

        x, y = get_data()
        steps = np.linspace(1, x.shape[1], x.shape[1])
        h1 = None
        h2 = None
        plt.cla()

        for step in range(int(x.shape[1] / TIME_STEP)):
            step_x = x[:, TIME_STEP*step:TIME_STEP*(step+1), :]
            step_y = y[:, TIME_STEP*step:TIME_STEP*(step+1), :]

            step_x = torch.from_numpy(step_x).type(torch.FloatTensor)
            step_y = torch.from_numpy(step_y).type(torch.FloatTensor)

            prediction, (h1, h2) = rnn(step_x, h1, h2)

            loss = loss_func(prediction, step_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # plt.subplot(211)
            # plt.cla()
            plt.plot(steps[TIME_STEP*step:TIME_STEP*(step+1)], step_y.data.numpy()[0, :, 0].flatten(), 'r-')

            # plt.subplot(212)
            # plt.cla()
            plt.plot(steps[TIME_STEP*step:TIME_STEP*(step+1)], prediction.data.numpy()[0, :, 0].flatten(), 'b-')

            plt.draw()
            # plt.pause(0.05)

    plt.ioff()
    plt.show()


def test():
    torch.autograd = False

    x, y = get_data()
    steps = np.linspace(1, x.shape[1], x.shape[1])

    start = 20
    end = 1000

    x = x[:, start:end, :]
    y = y[:, start:end, :]

    h1 = None
    h2 = None

    for step in range(int((end - start) / TIME_STEP)):
        step_x = x[:, TIME_STEP*step:TIME_STEP*(step+1), :]
        step_y = y[:, TIME_STEP*step:TIME_STEP*(step+1), :]

        step_x = torch.from_numpy(step_x).type(torch.FloatTensor)
        step_y = torch.from_numpy(step_y).type(torch.FloatTensor)

        prediction, (h1, h2) = rnn(step_x, h1, h2)

        plt.plot(steps[TIME_STEP*step:TIME_STEP*(step+1)], step_y.data.numpy()[0, :, 0].flatten(), 'r-')
        plt.plot(steps[TIME_STEP*step:TIME_STEP*(step+1)], prediction.data.numpy()[0, :, 0].flatten(), 'b-')

        plt.draw()
        plt.pause(0.05)
    plt.ioff()
    plt.show()


def get_data():
    path = mi_path
    csv_files = os.listdir(path)
    df = pd.read_csv(path + '/' + csv_files[0])

    mag = np.array([list(df['mag_x']), list(df['mag_y']), list(df['mag_z'])])
    y_coor = np.linspace(13, 140, mag.shape[1])
    x = np.zeros((mag.shape[1], INPUT_SIZE))
    y = np.zeros((mag.shape[1], OUTPUT_SIZE))
    for i in range(mag.shape[1]):
        x[i] = [mag[0][i], mag[1][i], mag[2][i]]
        y[i] = [y_coor[i], 0]

    x = x.reshape(1, x.shape[0], x.shape[1])
    y = y.reshape(1, y.shape[0], y.shape[1])
    return x, y

    # train_data = torchvision.datasets.MNIST(
    #     root=MNIST_ROOT,
    #     train=True,
    #     transform=torchvision.transforms.ToTensor(),
    #     download=False
    # )
    # train_loader = data.DataLoader(
    #     dataset=train_data,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False
    # )


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

if __name__ == '__main__':
    # train()
    save_restore(False)
    test()

    # x, y = get_data()
    # y = y[0, :, 0]
    # print(x.shape)
    # print(y.shape)
    # print(y)
