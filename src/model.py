import torch
from torch import nn
import torch.distributions as D

from typing import Tuple
import math


class PopGenerator(nn.Module):
    def __init__(
        self, population_size: int, genome_size: int, zdim: int, *args, **kwargs
    ):
        """
        Constructor for popultion generator. population_size, genome_size and zdim have to match with rest of the code

        :param population_size: size of populaion
        :type population_size: int
        :param genome_size: size of one genome
        :typr genome_size: int
        :param zdim: size of latent space
        :type zdim: int
        """
        super(PopGenerator, self).__init__(*args, **kwargs)

        self.zdim = zdim
        self.gen_size = genome_size
        self.pop_size = population_size

        self.net = nn.Sequential(
            nn.Linear(zdim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 2 * genome_size),
        )

        self.mask_encoder = nn.Sequential(
            nn.Linear(zdim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, genome_size),
        )

    def sample(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples given latent space z

        :param z: latent tensor
        :type z: torch.Tensor
        :return: tuple of generated exponents, log of probability of generated exponents, generated mask logits, and log of probability of generated mask logits
        :rtype: Tuple[Tensor, Tensor, Tensor, Tensor]
        """
        out = self.net(z)
        mu, log_std = out.chunk(2, dim=1)
        log_std = log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)

        eps = torch.randn_like(std)
        x = mu + eps * std
        x_det = x.detach()
        logp_per_dim = -0.5 * (
            ((x_det - mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        logp = logp_per_dim.sum(dim=1)

        logits = self.mask_encoder(z)
        probs = torch.sigmoid(logits)
        b = D.Bernoulli(probs=probs)
        m_sample = b.sample()
        logp_m = b.log_prob(m_sample).sum(dim=1)

        m_st = m_sample + probs - probs.detach()

        logp = logp + logp_m
        return x, logp, log_std, m_st

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Wrapper for sample - used only for use when you don't want to train the net

        :param z: latent tensor
        :type z: torch.Tensor
        :return: tuple of exponends and mask
        :rtype: Tuple[Tensor, Tensor]
        """
        x, _, _, m = self.sample(z)
        return x, m


class GaussianPolicy(nn.Module):
    def __init__(self, pop_size: int, zdim: int):
        """
        Constructor for latent space. population_size and zdim have to match with rest of the code

        :param pop_size: size of populations
        :type pop_size: int
        :param zdim: size of latent space
        :type zdim: int
        """
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(pop_size, zdim))
        self.log_std = nn.Parameter(torch.zeros(pop_size, zdim))

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples latent space

        :return: tuple of latent and log probability of latent
        :rtype: Tuple[Tensor, Tensor]
        """
        log_std = self.log_std.clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        x = self.mu + eps * std
        x_det = x.detach()
        logp_per_dim = -0.5 * (
            ((x_det - self.mu) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        logp = logp_per_dim.sum(dim=1)
        return x, logp
