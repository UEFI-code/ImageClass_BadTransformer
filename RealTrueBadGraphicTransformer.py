import torch
import torch.nn as nn

class BadGraphTransformerDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), deepth = 2, debug=False):
        super(BadGraphTransformerDown, self).__init__()
        self.convEncodingGroupA = nn.Sequential()
        self.convDecodingGroup = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.convEncodingGroupA.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupA.append(activation)
            else:
                self.convEncodingGroupA.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
                self.convEncodingGroupA.append(activation)
            self.convDecodingGroup.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
            self.convDecodingGroup.append(activation)
        self.normalization = normalization
        self.debug = debug
    
    def forward(self, x):
        a = self.convEncodingGroupA(x)
        x = torch.matmul(a.transpose(2, 3), a) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: xSqure shape {x.shape}')
        x = torch.matmul(a, x)
        x = self.convDecodingGroup(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x

class BadGraphTransformerUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride=1, padding=0, normalization = None, activation = nn.ReLU(), deepth = 2, debug=False):
        super(BadGraphTransformerUp, self).__init__()
        self.transConvEncodingGroupA = nn.Sequential()
        self.transConvDecodingGroup = nn.Sequential()
        for i in range(deepth):
            if i == 0:
                self.transConvEncodingGroupA.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupA.append(activation)
            else:
                self.transConvEncodingGroupA.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
                self.transConvEncodingGroupA.append(activation)
            self.transConvDecodingGroup.append(nn.ConvTranspose2d(out_channels, out_channels, kernel_size, stride, padding))
            self.transConvDecodingGroup.append(activation)
        self.normalization = normalization
        self.debug = debug

    def forward(self, x):
        a = self.transConvEncodingGroupA(x)
        x = torch.matmul(a.transpose(2, 3), a) # Here is to semantic hybrid.
        if self.debug:
            print(f'Debug: xSqure shape {x.shape}')
        x = torch.matmul(a, x)
        x = self.transConvDecodingGroup(x)
        if self.normalization is not None:
            x = self.normalization(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 64, 64)
    badGraphTransDown = BadGraphTransformerDown(3, 16, debug=True)
    x = badGraphTransDown(x)
    print(f'After badGraphTransDown: {x.shape}')
    badGraphTransUp = BadGraphTransformerUp(16, 16, debug=True)
    x = badGraphTransUp(x)
    print(f'After badGraphTransUp: {x.shape}')