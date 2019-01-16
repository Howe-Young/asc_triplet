"""
some networks

"""
import torch.nn as nn
import torch.nn.functional as F
import torch


# TODO better network(cnn and lstm)
class vggish_bn(nn.Module):

    def __init__(self, classify=False):
        super(vggish_bn, self).__init__()

        self.classify = classify
        # convolution block 1
        self.conv1_1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1)
        self.bn1_1 = nn.BatchNorm2d(8)
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1)
        self.bn1_2 = nn.BatchNorm2d(8)
        self.pool1_1 = nn.MaxPool2d(kernel_size=(2, 2))

        # convolution block 2
        self.conv2_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn2_1 = nn.BatchNorm2d(16)
        self.conv2_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.bn2_2 = nn.BatchNorm2d(16)
        self.pool2_1 = nn.MaxPool2d(kernel_size=(2, 2))

        # convolution block 3
        self.conv3_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn3_1 = nn.BatchNorm2d(32)
        self.conv3_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn3_2 = nn.BatchNorm2d(32)
        self.pool3_1 = nn.MaxPool2d(kernel_size=(2, 2))

        # convolution block 4
        self.conv4_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn4_1 = nn.BatchNorm2d(64)
        self.conv4_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn4_2 = nn.BatchNorm2d(64)
        self.pool4_1 = nn.MaxPool2d(kernel_size=(2, 2))

        # full connect layer
        self.lstm = nn.LSTM(input_size=64 * 2, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(128, 64)
        self.cls = nn.Linear(64, 10)

    def forward(self, x):
        # block 1
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = F.relu(x)
        x = self.pool1_1(x)

        # block 2
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = F.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = F.relu(x)
        x = self.pool2_1(x)

        # block 3
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = F.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = F.relu(x)
        x = self.pool3_1(x)

        # block 4
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = F.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = F.relu(x)
        x = self.pool4_1(x)

        # full connect layer
        x = x.view(-1, 64 * 2, 31)
        x = x.permute(0, 2, 1)
        out, hidden = self.lstm(x, None) # out shape (batch_size, time_step, hidden_size * num_directions)
        out = torch.mean(out, dim=1) # out shape (batch_size, hidden_size * num_directions)
        x = self.fc(out)

        # pretrain classification network
        if self.classify:
            x = F.relu(x)
            x = self.cls(x)

        return x

    def get_embeddings(self, x):
        return self.forward(x)

    def set_classify(self, classifiy):
        self.classify = classifiy


class embedding_net_shallow(nn.Module):
    def __init__(self):
        super(embedding_net_shallow, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        # self.pool3 = nn.MaxPool2d(kernel_size=(10, 125))
        # input_size is frequency, the filters number is 32, concat along the frequency axis.
        self.lstm1 = nn.LSTM(input_size=32 * 10, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 32 * 10, 125)
        # lstm input is (batch, time_step, input_dims)
        x = x.permute(0, 2, 1)
        out, hidden = self.lstm1(x, None)
        # out shape is: (batch, seq_length, num_directions * hidden_size)   seq_length is time_step
        # hidden is a tuple, include (h_n, c_n)
        # h_n shape is: (batch, num_layers * num_directions, hidden_size)
        # c_n shape is: (batch, num_layers * num_directions, hidden_size)
        # return torch.cat((out[:, 0, :], out[:, -1, :]), dim=1) # return first time step concat last time step
        return torch.mean(out, dim=1)

    def get_embeddings(self, x):
        return self.forward(x)


class classifier(nn.Module):
    def __init__(self):
        super(classifier, self).__init__()
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.fc(x)
        return x


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5), padding=2, stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(num_features=8)

        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2)
        self.bn3 = nn.BatchNorm2d(num_features=16)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), padding=2, stride=(2, 2))
        self.bn4 = nn.BatchNorm2d(num_features=16)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2)
        self.bn5 = nn.BatchNorm2d(num_features=32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2, stride=(2, 2))
        self.bn6 = nn.BatchNorm2d(num_features=32)

        self.gru = nn.GRU(160, 64, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=128, out_features=64)

        nn.init.orthogonal_(self.gru.weight_ih_l0); nn.init.constant_(self.gru.bias_ih_l0, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0); nn.init.constant_(self.gru.bias_hh_l0, 0)
        nn.init.orthogonal_(self.gru.weight_ih_l0_reverse); nn.init.constant_(self.gru.bias_ih_l0_reverse, 0)
        nn.init.orthogonal_(self.gru.weight_hh_l0_reverse); nn.init.constant_(self.gru.bias_hh_l0_reverse, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))     #(batch, 32, 5, 63)
        x = x.view(-1, 32 * 5, 63)              #(B, C*F, T)
        x = x.permute(0, 2, 1)                  #(B, T, C*F)
        x, _ = self.gru(x)                      #(B, T, 128)
        x = torch.mean(x, dim=1)                #(B, 128)
        x = self.fc(x)                          #(B, 64)
        return x

    def get_embeddings(self, x):
        return self.forward(x)


