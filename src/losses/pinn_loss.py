import torch.nn as nn


class twsa_loss(nn.Module):
    """ dTWS/dt = dP/dt - dET/dt-dQ/dt
    dTWS/dt = TWSA
    dP/dt = P(t+1)-P(t)/dt 
    """
    def __init__(self, pinn_loss):
        super(twsa_loss, self).__init__()
        self.pinn_loss = pinn_loss

    def forward(self, twsa_pred, u_target, f_pred):
        return self.pinn_loss(u_pred, u_target, f_pred)