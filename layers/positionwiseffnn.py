import torch.nn as nn
import torch
class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        return x



if __name__=='__main__':
    x = torch.ones([1,100,512])
    print(x.size())
    pwffn = PositionWiseFeedForwardNN(512, 100)
    print(pwffn(x).size())