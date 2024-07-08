import torch.nn as nn

class VanillaEncoder(nn.Module):  # TODO unit tests!
    def __init__(self, datalen, embedding_size, numhidden, numconvs, numfilters, kernel_size, stride=1, pad=True,
                 device='cuda'):
        super().__init__()
        # One data example comes in as 1xn -> 1xnx2, so whole will come in m x n x 2
        self.permutation = nn.Linear(datalen, datalen, bias=False).to(device)
        self.convs = [nn.Conv1d(2, numfilters, kernel_size, stride,
                                padding=kernel_size-1 if pad else 'valid').to(device)]
        self.decode_convs = [nn.ConvTranspose1d(numfilters, 2, kernel_size, stride,
                                                padding=kernel_size-1 if pad else 'valid').to(device)]
        # Complex values in second channel
        self.pools = [nn.MaxPool1d(2, stride=stride).to(device) for _ in range(numconvs)]
        self.unpools = [nn.MaxUnpool1d(2, stride=stride).to(device) for _ in range(numconvs)]
        self.hiddens = []  # Records the input sizes, since embedding size is fixed
        self.decode_hiddens = []
        self.unpermute = nn.Linear(datalen, datalen, bias=False).to(device)

        stepdown = (datalen - embedding_size) // numhidden
        c = numfilters
        for i in range(numconvs-1):
            self.convs.append(nn.Conv1d(c, 2*c, kernel_size, stride,
                                        padding=kernel_size-1 if pad else 'valid').to(device))
            self.decode_convs = ([nn.ConvTranspose1d(2*c, c, kernel_size, stride,
                                                    padding=kernel_size-1 if pad else 'valid').to(device)]
                                 + self.decode_convs)
            c *= 2
        while c - stepdown > embedding_size:
            self.hiddens.append(nn.Linear(c, c-stepdown).to(device))
            self.decode_hiddens = [nn.Linear(c-stepdown, c).to(device)] + self.decode_hiddens
            c -= stepdown
        self.hiddens.append(nn.Linear(c, embedding_size).to(device))
        self.decode_hiddens = [nn.Linear(embedding_size, c).to(device)] + self.decode_hiddens

        self.activations = [nn.ReLU().to(device) for _ in range(len(self.hiddens)-1)]
        self.activations.append(nn.Sigmoid().to(device))
        self.decode_activations = [nn.ReLU().to(device) for _ in range(len(self.hiddens))]

    def forward_encode(self, x):  # Returns an embedding
        permute = self.permutation(x)
        for i in range(len(self.convs)):
            permute = self.pools[i](self.convs[i](permute))
        for i in range(len(self.hiddens)):
            permute = self.activations[i](self.hiddens[i](permute))
        # Last activation is Sigmoid, so it clamps.
        return permute

    def forward_decode(self, embedding):  # Returns original vector
        x = self.decode_hiddens[0](embedding)  # Skip the activation to unclamp
        for i in range(1, len(self.decode_hiddens)):
            x = self.decode_activations[i](self.decode_hiddens[i](x))
        for i in range(len(self.decode_convs)):
            x = self.unpools[i](self.decode_convs[i](x))
        vec = self.unpermute(x)
        return vec
