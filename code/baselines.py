
import torch
import torch.nn as nn
import torch.nn.functional as F

# %% Deep convnet - Baseline 1
class deepConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(25, 25, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(inF, outF, kernalSize, max_norm=0.5, *args, **kwargs))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
        super(deepConvNet, self).__init__()

        kernalSize = (1, 5)   # Please note that the kernel size in the origianl paper is (1, 10), we found when the segment length is shorter than 2s larger kernel size will 
                              # cause network error. Besides using (1, 5) when EEG segment is 4s gives slightly higher ACC and F1 with a smaller model size. 
        nFilt_FirstLayer = 25
        nFiltLaterLayer = [25, 50, 100, 200]

        firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        middleLayers = nn.Sequential(*[self.convBlock(inF, outF, dropoutP, kernalSize)
                                       for inF, outF in zip(nFiltLaterLayer, nFiltLaterLayer[1:])])

        self.allButLastLayers = nn.Sequential(firstLayer, middleLayers)

        self.fSize = self.calculateOutSize(self.allButLastLayers, nChan, nTime)
        self.lastLayer = self.lastBlock(nFiltLaterLayer[-1], nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.allButLastLayers(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x


# %% Shallow convnet - Baseline 3
class shallowConvNet(nn.Module):
    def convBlock(self, inF, outF, dropoutP, kernalSize, *args, **kwargs):
        return nn.Sequential(
            nn.Dropout(p=dropoutP),
            Conv2dWithConstraint(inF, outF, kernalSize, bias=False, max_norm=2, *args, **kwargs),
            nn.BatchNorm2d(outF),
            nn.ELU(),
            nn.MaxPool2d((1, 3), stride=(1, 3))
        )

    def firstBlock(self, outF, dropoutP, kernalSize, nChan, *args, **kwargs):
        return nn.Sequential(
            Conv2dWithConstraint(1, outF, kernalSize, padding=0, max_norm=2, *args, **kwargs),
            Conv2dWithConstraint(40, 40, (nChan, 1), padding=0, bias=False, max_norm=2),
            nn.BatchNorm2d(outF),
        )

    def calculateOutSize(self, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        block_one = self.firstLayer
        avg = self.avgpool
        dp = self.dp
        out = torch.log(block_one(data).pow(2))
        out = avg(out)
        out = dp(out)
        out = out.view(out.size()[0], -1)
        return out.size()

    def __init__(self, nChan, nTime, nClass=2, dropoutP=0.25, *args, **kwargs):
        super(shallowConvNet, self).__init__()

        kernalSize = (1, 25)
        nFilt_FirstLayer = 40

        self.firstLayer = self.firstBlock(nFilt_FirstLayer, dropoutP, kernalSize, nChan)
        self.avgpool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dp = nn.Dropout(p=dropoutP)
        self.fSize = self.calculateOutSize(nChan, nTime)
        self.lastLayer = nn.Linear(self.fSize[-1], nClass)

    def forward(self, x):
        x = self.firstLayer(x)
        x = torch.log(self.avgpool(x.pow(2)))
        x = self.dp(x)
        x = x.view(x.size()[0], -1)
        x = self.lastLayer(x)

        return x


# %% EEGNet Baseline 2
class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding=(0, self.C1 // 2), bias=False),
                nn.BatchNorm2d(self.F1),
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1),
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4),
                nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22),
                                     padding=(0, 22//2), bias=False,
                                     groups=self.F1 * self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1),
                          stride=1, bias=False, padding=0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8),
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs))

    def calculateOutSize(self, model, nChan, nTime):
        '''
        Calculate the output based on input size.
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        out = model(data).shape
        return out[2:]

    def __init__(self, nChan, nTime, nClass=2,
                 dropoutP=0.25, F1=8, D=2,
                 C1=64, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1
        self.F1 = F1
        self.D = D
        self.nTime = nTime
        self.nClass = nClass
        self.nChan = nChan
        self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = self.firstBlocks(x)
        x = self.lastLayer(x)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 2)

        return x


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)
