import torch
from torch import nn


class NeuralDensityNetwork(nn.Module):
    def __init__(
        self, feature_extractor: nn.Module, density_estimator: nn.Module,
    ):
        super(NeuralDensityNetwork, self).__init__()
        self.op = nn.Sequential(feature_extractor, density_estimator,)

    def forward(self, inputs):
        return self.op(inputs)


class TarNet(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        hypothesis_0: nn.Module,
        hypothesis_1: nn.Module,
        density_estimator: nn.Module,
    ):
        super(TarNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.hypothesis_0 = hypothesis_0
        self.hypothesis_1 = hypothesis_1
        self.density_estimator = density_estimator

    def forward(self, inputs):
        x, t = inputs
        phi = self.feature_extractor(x)
        phi = (1 - t) * self.hypothesis_0(phi) + t * self.hypothesis_1(phi)
        return self.density_estimator([phi, t])
