import torch
import torch.nn as nn
import bad_graph_transformer
import RealBadGraphicTransformer
import RealTrueBadGraphicTransformer

class myCVModel(nn.Module):
    def __init__(self, arch = 'BadGraphicTransformer', debug = False):
        super().__init__()
        if arch == 'BadGraphicTransformer':
            self.encoder = bad_graph_transformer.BadGraphTransformerDown
        elif arch == 'RealBadGraphicTransformer':
            self.encoder = RealBadGraphicTransformer.BadGraphTransformerDown
        elif arch == 'RealTrueBadGraphicTransformer':
            self.encoder = RealTrueBadGraphicTransformer.BadGraphTransformerDown
        else:
            raise ValueError(f'Arch {arch} not supported')
        # Let's play
        self.encoderGroup = nn.Sequential()
        self.encoderGroup.append(self.encoder(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=0, normalization=nn.BatchNorm2d(64), activation = nn.ReLU(), deepth = 1, debug=debug))
        self.encoderGroup.append(self.encoder(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0, normalization=nn.BatchNorm2d(64), activation = nn.ReLU(), deepth = 1, debug=debug))
        self.encoderGroup.append(self.encoder(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, normalization=nn.BatchNorm2d(64), activation = nn.ReLU(), deepth = 1, debug=debug))

        self.decoderGroup = nn.Sequential()
        self.decoderGroup.append(nn.Linear(64 * 144, 4096))
        self.decoderGroup.append(nn.ReLU())
    
    def forward(self, x):
        x = self.encoderGroup(x)
        x = x.view(x.size(0), -1)
        x = self.decoderGroup(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = myCVModel(arch = 'BadGraphicTransformer', debug = True)
    print(model(x).shape)