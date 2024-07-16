import torch.nn as nn
import torch

class VanillaAutoEncoder(nn.Module):  # TODO unit tests!
    def __init__(self, datalen, embedding_size, numhidden, numconvs, numfilters, kernel_size, stride=1, pad=True,
                 device='cuda'):
        super().__init__()
        # One data example comes in as 1xn -> 1xnx2, so whole will come in m x n x 2
        self.permutation = nn.Linear(datalen, datalen, bias=False).to(device)  # TODO want to force permutation in loss
        self.convs = [nn.Conv1d(2, numfilters, kernel_size, stride,
                                padding=kernel_size-1 if pad else 'valid').to(device)]
        self.decode_convs = [nn.ConvTranspose1d(numfilters, 2, kernel_size, stride,
                                                padding=kernel_size-1 if pad else 'valid').to(device)]
        # Complex values in second channel
        self.pools = [nn.MaxPool1d(2, stride=stride).to(device) for _ in range(numconvs)]
        self.unpools = [nn.MaxUnpool1d(2, stride=stride).to(device) for _ in range(numconvs)]
        self.hiddens = []  # Records the input sizes, since embedding size is fixed
        self.decode_hiddens = []
        self.unpermute = nn.Linear(datalen, datalen, bias=False).to(device)  # TODO force permutation via loss

        self.stepdown = (datalen - embedding_size) // numhidden
        c = numfilters
        for i in range(numconvs-1):
            self.convs.append(nn.Conv1d(c, 2*c, kernel_size, stride,
                                        padding=kernel_size-1 if pad else 'valid').to(device))
            self.decode_convs = ([nn.ConvTranspose1d(2*c, c, kernel_size, stride,
                                                    padding=kernel_size-1 if pad else 'valid').to(device)]
                                 + self.decode_convs)
            c *= 2
        self._final_channels = c
        while c - self.stepdown > embedding_size:
            self.hiddens.append(nn.Linear(c, c-self.stepdown).to(device))
            self.decode_hiddens = [nn.Linear(c-self.stepdown, c).to(device)] + self.decode_hiddens
            c -= self.stepdown
        self.hiddens.append(nn.Linear(c, embedding_size).to(device))
        self.decode_hiddens = [nn.Linear(embedding_size, c).to(device)] + self.decode_hiddens

        self.activations = [nn.ReLU().to(device) for _ in range(len(self.hiddens)-1)]
        self.activations.append(nn.Sigmoid().to(device))
        self.decode_activations = [nn.ReLU().to(device) for _ in range(len(self.hiddens))]

    def __forward_conv(self, x):
        # TODO separate complex -> R^2
        permute = self.permutation(x)
        # TODO reshape from mxnx2 to mx1xnx2
        for i in range(len(self.convs)):
            permute = self.pools[i](self.convs[i](permute))
        return permute

    def __forward_linear(self, x):
        y = x
        for i in range(len(self.hiddens)):
            y = self.activations[i](self.hiddens[i](y))
        return y

    def forward_encode(self, x):  # Returns an embedding
        y = self.__forward_linear(self.__forward_conv(x))
        return y

    def forward_decode(self, embedding):  # Returns original vector
        x = self.decode_hiddens[0](embedding)  # Skip the activation to unclamp
        for i in range(1, len(self.decode_hiddens)):
            x = self.decode_activations[i](self.decode_hiddens[i](x))
        for i in range(len(self.decode_convs)):
            x = self.unpools[i](self.decode_convs[i](x))
        vec = self.unpermute(x)
        return vec

    def forward(self, x):
        s1 = self.forward_encode(x)
        s2 = self.forward_decode(s1)
        return s2

    def __reconstruction_loss(self, x_hat, x):
        reconstruction_loss = torch.F.binary_cross_entropy(x_hat, x, reduction='sum')
        return reconstruction_loss

    def loss(self, x):
        return self.__reconstruction_loss(self.forward(x), x)  # TODO check if needs to be scaled?


class VariationalAutoEncoder(VanillaAutoEncoder):
    def __init__(self, datalen, embedding_size, numhidden, numconvs, numfilters, kernel_size, stride=1, pad=True,
                 device='cuda'):
        super().__init__(datalen, embedding_size, numhidden, numconvs, numfilters, kernel_size, stride, pad, device)
        self.varhidden = []
        c = self._final_channels
        while c - self.stepdown > embedding_size:
            self.varhidden.append(nn.Linear(c, c-self.stepdown))
            c -= self.stepdown
        self.varhidden.append(nn.Linear(c, embedding_size))


    def reparameterize(self, mu, log_var):
        epsilon = torch.randn_like(log_var)
        z = torch.exp(.5 * log_var) * epsilon + mu
        return z

    def forward_encode(self, x):
        logvar = self.__forward_conv(x)
        mu = super().__forward_linear(logvar)
        for i in range(len(self.varhidden)):
            logvar = self.activations[i](self.varhidden[i](logvar))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        s1, mu, logvar = self.forward_encode(x)
        s2 = self.forward_encode(s1)
        return s2, mu, logvar

    def loss(self, x):
        x_hat, mu, logvar = self.forward(x)
        reconstruction_loss = self.__reconstruction_loss(x_hat, x)
        kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = (reconstruction_loss + kl_divergence_loss) / x.size(0)
        return loss
