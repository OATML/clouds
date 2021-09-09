import torch
from torch import distributions, nn


class Normal(nn.Module):
    def __init__(
        self, dim_input, dim_output,
    ):
        super(Normal, self).__init__()
        self.mu = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        sigma = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        self.sigma = nn.Sequential(sigma, nn.Softplus())

    def forward(self, inputs):
        return distributions.Normal(
            loc=self.mu(inputs), scale=self.sigma(inputs) + 1e-7
        )


class SplitNormal(nn.Module):
    def __init__(
        self, dim_input, dim_output,
    ):
        super(SplitNormal, self).__init__()
        self.mu0 = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        sigma0 = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        self.sigma0 = nn.Sequential(sigma0, nn.Softplus())
        self.mu1 = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        sigma1 = nn.Linear(in_features=dim_input, out_features=dim_output, bias=True,)
        self.sigma1 = nn.Sequential(sigma1, nn.Softplus())

    def forward(self, inputs):
        x, t = inputs
        loc = (1 - t) * self.mu0(x) + t * self.mu1(x)
        scale = (1 - t) * self.sigma0(x) + t * self.sigma1(x) + 1e-7
        return distributions.Normal(loc=loc, scale=scale)


class MixtureSameFamily(distributions.MixtureSameFamily):
    def log_prob(self, inputs):
        loss = torch.exp(self.component_distribution.log_prob(inputs.unsqueeze(1)))
        loss = torch.sum(loss * self.mixture_distribution.probs, dim=1)
        return torch.log(loss + 1e-7)


class GMM(nn.Module):
    def __init__(
        self, num_components, dim_input, dim_output,
    ):
        super(GMM, self).__init__()
        self.mu = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        sigma = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        self.pi = nn.Linear(
            in_features=dim_input, out_features=num_components, bias=True,
        )
        self.sigma = nn.Sequential(sigma, nn.Softplus())
        self.num_components = num_components
        self.dim_output = dim_output

    def forward(self, inputs):
        loc = self.mu(inputs).reshape(-1, self.num_components, self.dim_output)
        scale = (
            self.sigma(inputs).reshape(-1, self.num_components, self.dim_output) + 1e-7
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=self.pi(inputs)),
            component_distribution=distributions.Independent(
                base_distribution=distributions.Normal(loc=loc, scale=scale,),
                reinterpreted_batch_ndims=1,
            ),
        )


class SplitGMM(nn.Module):
    def __init__(
        self, num_components, dim_input, dim_output,
    ):
        super(SplitGMM, self).__init__()
        self.mu0 = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        sigma0 = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        self.pi0 = nn.Linear(
            in_features=dim_input, out_features=num_components, bias=True,
        )
        self.sigma0 = nn.Sequential(sigma0, nn.Softplus())
        self.mu1 = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        sigma1 = nn.Linear(
            in_features=dim_input, out_features=num_components * dim_output, bias=True,
        )
        self.pi1 = nn.Linear(
            in_features=dim_input, out_features=num_components, bias=True,
        )
        self.sigma1 = nn.Sequential(sigma1, nn.Softplus())
        self.dim_output = dim_output
        self.num_components = num_components

    def forward(self, inputs):
        x, t = inputs
        logits = (1 - t) * self.pi0(x) + t * self.pi1(x)
        loc = (1 - t) * self.mu0(x) + t * self.mu1(x)
        scale = (1 - t) * self.sigma0(x) + t * self.sigma1(x) + 1e-7
        component_distribution = distributions.Independent(
            distributions.Normal(
                loc=loc.reshape(-1, self.num_components, self.dim_output),
                scale=scale.reshape(-1, self.num_components, self.dim_output),
            ),
            reinterpreted_batch_ndims=1,
        )
        return MixtureSameFamily(
            mixture_distribution=distributions.Categorical(logits=logits),
            component_distribution=component_distribution,
        )


class Categorical(nn.Module):
    def __init__(
        self, dim_input, dim_output,
    ):
        super(Categorical, self).__init__()
        self.logits = nn.Linear(
            in_features=dim_input, out_features=dim_output, bias=True,
        )
        self.distribution = (
            distributions.Bernoulli if dim_output == 1 else distributions.Categorical
        )

    def forward(self, inputs):
        return self.distribution(logits=self.logits(inputs))
