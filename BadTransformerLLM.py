import torch
import torch.nn as nn
class myBadTransfomerBlock(nn.Module):
    def __init__(self, dim=4096):
        super().__init__()
        self.li1 = nn.Linear(dim, dim, bias=False)
        self.li2 = nn.Linear(dim, dim, bias=False)
        self.li3 = nn.Linear(dim, dim, bias=False)
        self.li4 = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        xA, xB, xC = self.li1(x), self.li2(x), self.li3(x)
        theShape = xA.shape
        theDim = len(theShape)
        xSqure = torch.matmul(xA.transpose(theDim - 2, theDim - 1), xB)
        xSqure = torch.nn.functional.softmax(xSqure, dim=-1)
        xO = torch.matmul(xC, xSqure)
        xO = torch.nn.functional.softmax(xO, dim=-1)
        xO = self.li4(xO)
        xO = torch.nn.functional.silu(xO)
        xO = self.out(x + xO)
        return xO

class myBadTransformerUnit(nn.Module):
    def __init__(self, num_layers=2, process_dim=4096):
        super().__init__()
        self.badtrans = []
        for _ in range(num_layers):
            self.badtrans.append(myBadTransfomerBlock(dim=process_dim))
        self.badtrans = nn.ModuleList(self.badtrans)

    def forward(self, x):
        for badtrans in self.badtrans:
            x = badtrans(x)
        return x

class myCVModel(nn.Module):
    def __init__(self, num_layers=6, process_dim=1024):
        super().__init__()
        self.encoder = nn.Linear(224, process_dim)
        self.badtrans = myBadTransformerUnit(num_layers=num_layers, process_dim=process_dim)
        self.out = nn.Linear(process_dim, 1000)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.badtrans(x)[:,-1,-1]
        x = self.out(x)
        x = self.softmax(x)
        return x
