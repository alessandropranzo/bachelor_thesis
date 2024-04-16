import torch

def diffusion(x: torch.Tensor, t: torch.Tensor):
    """Returns the analytical solution u(x,t) = sin(pi * x) * e^(-t) to the diffusion problem described by du/dt = d2u/d2x + e^(-t)(-sin(pi*x) + pi^2 * sin(pi*x)) with boundary conditions u(x, 0) = sin(pi*x) and initial value u(-1, t) = u(1, t) = 0 .

        Inputs:
        - x: torch.Tensor = the position variable
        - t: torch.Tensor = the time variable
    """
    out = torch.sin(torch.pi * x) * torch.exp(-t)
    return out


def diff_loss():
    pass


def inversed_diffusion_reaction(x: torch.Tensor):
    """ Returns the analytical solution k(x) = 0.1 + e^[-0.5 * (x - 0.5)^2 / (0.15)^2] to the diffusion reaction inverse problem described by 0.01 * d2u/d2x - k(x)u = sin(2*pi*x).

        Inputs:
        - x: torch.Tensor = the position varible
    """
    a = torch.Tensor(0.1)
    b = torch.Tensor(-0.5)
    num = (x + b)**2
    den = (torch.Tensor(0.15))**2
    out = a + torch.exp(b * (num/den))
    return out


def inv_diff_react_loss():
    pass