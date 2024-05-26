import torch
import torch.nn as nn
from torch.nn import init
class Conv3DLSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Conv3DLSTMModel, self).__init__()

        self.conv1 = nn.Conv3d(1, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv3 = nn.Conv3d(256, 75, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.lstm1 = nn.LSTM(input_size=75 * 5 * 17, hidden_size=hidden_size,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        self.lstm2 = nn.LSTM(input_size=128 * 2, hidden_size=hidden_size,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        self.dense = nn.Linear(128 * 2, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Apply the sequence of conv, relu activations and max pooling
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool3(x)
        # Flatten the dimensions other than batch and sequence length (depth)
        batch_size, _, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  # Swap the depth and channel dimensions
        x = x.reshape(batch_size, D, -1)  # Flatten the spatial dimensions
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # To apply the dense layer, we need to consider only the last output of the sequence.
        x = self.dense(x)

        return x
    
class Conv3DLSTMModelMini(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(Conv3DLSTMModelMini, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))

        self.lstm1 = nn.LSTM(input_size=128 * 11 * 35, hidden_size=hidden_size,
                              batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.3)

        self.dense = nn.Linear(64 * 2, vocab_size)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        batch_size, _, D, H, W = x.size()
        x = x.permute(0, 2, 1, 3, 4)  
        x = x.reshape(batch_size, D, -1)  # Flatten the spatial dimensions

        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x = self.dense(x)

        return x

class LipNet(nn.Module):
    def __init__(self, vocab_size=40, hidden_size=256):
        super(LipNet, self).__init__()
        # Adjustments for the number of initial channels if needed
        self.conv1 = nn.Conv3d(1, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
        self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
        self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))     
        self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        
        # Adjust the input size according to the output from the last conv layer
        self.lstm1 = nn.LSTM(96*2*8, hidden_size, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, hidden_size, 1, batch_first=True, bidirectional=True)
        
        self.dense = nn.Linear(512, vocab_size)
        self.dropout_p = 0.5

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.dropout_p)
        self.dropout3d = nn.Dropout3d(self.dropout_p)  
        self._init_weights_()
    
    def _init_weights_(self):
        
        init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        init.constant_(self.conv1.bias, 0)
        
        init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        init.constant_(self.conv2.bias, 0)
        
        init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
        init.constant_(self.conv3.bias, 0)        
        
        init.kaiming_normal_(self.dense.weight, nonlinearity='sigmoid')
        init.constant_(self.dense.bias, 0)
        
        # Initialization for LSTM weights/biases can be more complex
        # Here is a simple version which you can refine depending on your needs
        for lstm in (self.lstm1, self.lstm2):
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout3d(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout3d(x)        
        x = self.pool3(x)

        x = x.permute(0, 2, 1, 3, 4)  # B, T, C, H, W
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # Flatten the spatial dimensions
        
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)
                
    
        x = self.dense(x)
    
        return x
    

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(1, 3, 224, 224).to(device)
    model = LipNet().to(device)
    print(model(x).shape)