
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, input_size=7, hidden_size=32, output_size=1, seq_len=30):
        super(MyNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size * seq_len, self.output_size)

    def forward(self, input):
        out, _ = self.lstm(input)
        b, s, h = out.size()
        out = self.fc(out.reshape(b, s * h))
        return out

if __name__ =='__main__':

    net = MyNet()
    print(list(net.parameters()))
