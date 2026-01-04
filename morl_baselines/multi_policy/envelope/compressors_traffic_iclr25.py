import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.optim as optim
import pdb
from morl_baselines.common.weights import random_weights

## pytorch parametrization module
# https://discuss.pytorch.org/t/positive-weights/19701/8
class SoftplusParameterization(nn.Module):
    def forward(self, X):
        return nn.functional.softplus(X)

class RowSumToOneParameterization(nn.Module):
    def forward(self, X):
        return nn.functional.softmax(X, dim=1)

class ReLUParameterization(nn.Module):
    def forward(self, X):
        return nn.functional.relu(X)

# class AE_PositiveEnc(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int):
#         super(AE_PositiveEnc, self).__init__()
#
#         encoder_layer = nn.Linear(input_size, hidden_size)
#         parametrize.register_parametrization(encoder_layer, "weight", SoftplusParameterization())
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             encoder_layer
#         )
#
#         decoder_layer = nn.Linear(hidden_size, input_size)
#         parametrize.register_parametrization(decoder_layer, "weight", SoftplusParameterization())
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             decoder_layer
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
#     def compress(self, x):
#         x = self.encoder(x)
#         return x


# Define the Autoencoder architecture with positive Encoder but arbitrary decoder for compressing reward.
## Layer we may relax the condition of (i) linearity or (ii) positivity (e.g., softplus - c).

# class AE_PositiveEnc(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, relax_positive: bool, drp: float):
#         super(AE_PositiveEnc, self).__init__()
#
#         encoder_layer = nn.Linear(input_size, hidden_size, bias=True) # no bias
#
#         if not relax_positive:
#             parametrize.register_parametrization(encoder_layer, "weight", SoftplusParameterization())
#
#             with torch.no_grad():
#                 assert torch.all(encoder_layer.weight > 0)  # now all > 0
#                 # parametrize.register_parametrization(encoder_layer, "weight", RowSumToOneParameterization())
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             encoder_layer
#         )
#
#         # Decoder
#         # decoder_layer = nn.Linear(hidden_size, input_size, bias=True)
#         # if not relax_positive:
#         #     parametrize.register_parametrization(decoder_layer, "weight", SoftplusParameterization())
#         #
#         #     with torch.no_grad():
#         #         assert torch.all(decoder_layer.weight > 0)  # now all > 0
#
#
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, input_size * 2),
#             nn.ReLU(True),
#             nn.Dropout(drp),
#             nn.Linear(input_size * 2, input_size * 2),
#             nn.ReLU(True),
#             nn.Dropout(drp),
#             nn.Linear(input_size * 2, input_size) # default: bias=True
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#
#         return x
#
#     def compress(self, x):
#         x = self.encoder(x)
#         return x


class AE_PositiveEnc(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, relax_positive: bool, drp: float):
        super(AE_PositiveEnc, self).__init__()

        encoder_layer = nn.Linear(input_size, hidden_size, bias=False) # no bias. linear
        # encoder_layer.weight.data = torch.abs(encoder_layer.weight.data)
        encoder_layer.weight.data = torch.ones(hidden_size, input_size)/input_size

        if not relax_positive:
            parametrize.register_parametrization(encoder_layer, "weight", RowSumToOneParameterization())

            with torch.no_grad():
                assert torch.all(encoder_layer.weight > 0)  # now all > 0

        # Encoder
        self.encoder = nn.Sequential(
            encoder_layer
        )

        # Decoder
        # decoder_layer = nn.Linear(hidden_size, input_size, bias=True)
        # if not relax_positive:
        #     parametrize.register_parametrization(decoder_layer, "weight", SoftplusParameterization())
        #
        #     with torch.no_grad():
        #         assert torch.all(decoder_layer.weight > 0)  # now all > 0

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size * 2),
            nn.ReLU(True),
            nn.Dropout(drp),
            nn.Linear(input_size * 2, input_size * 2),
            nn.ReLU(True),
            nn.Dropout(drp),
            nn.Linear(input_size * 2, input_size) # default: bias=True
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compress(self, x):
        x = self.encoder(x)
        return x



# class AE_PositiveEnc_Only(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int, relax_positive: bool):
#         super(AE_PositiveEnc_Only, self).__init__()
#
#         encoder_layer = nn.Linear(input_size, hidden_size, bias=True)
#
#         if not relax_positive:
#             parametrize.register_parametrization(encoder_layer, "weight", SoftplusParameterization())
#
#             with torch.no_grad():
#                 assert torch.all(encoder_layer.weight > 0)  # now all > 0
#
#             # parametrize.register_parametrization(encoder_layer, "weight", RowSumToOneParameterization())
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             encoder_layer
#         )
#
#
#
#     def compress(self, x):
#         x = self.encoder(x)
#         return x

class AE_PositiveEnc_Only(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, relax_positive: bool):
        super(AE_PositiveEnc_Only, self).__init__()

        encoder_layer = nn.Linear(input_size, hidden_size, bias=False)  # no bias. linear
        # encoder_layer.weight.data = torch.abs(encoder_layer.weight.data)
        encoder_layer.weight.data = torch.ones(hidden_size, input_size) / input_size

        if not relax_positive:
            parametrize.register_parametrization(encoder_layer, "weight", RowSumToOneParameterization())

            with torch.no_grad():
                assert torch.all(encoder_layer.weight > 0)  # now all > 0

        # Encoder
        self.encoder = nn.Sequential(
            encoder_layer
        )

    def compress(self, x):
        x = self.encoder(x)
        return x


class NPCA(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(NPCA, self).__init__()

        encoder_layer = nn.Linear(input_size, hidden_size, bias=False) # no bias
        # nn.init.uniform_(encoder_layer.weight, a=0.01, b=0.1)
        # encoder_layer.weight.data = torch.abs(encoder_layer.weight.data)
        encoder_layer.weight.data = torch.ones(hidden_size, input_size) / input_size

        parametrize.register_parametrization(encoder_layer, "weight", ReLUParameterization())

        with torch.no_grad():
            assert torch.all(encoder_layer.weight >= 0)

        # Encoder
        self.encoder = nn.Sequential(
            encoder_layer
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class AE_standard(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super(AE_standard, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size * 2),
            nn.ReLU(True),
            nn.Linear(input_size * 2, input_size * 2),
            nn.ReLU(True),
            nn.Linear(input_size * 2, hidden_size)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size * 2),
            nn.ReLU(True),
            nn.Linear(input_size * 2, input_size * 2),
            nn.ReLU(True),
            nn.Linear(input_size * 2, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def compress(self, x):
        x = self.encoder(x)
        return x

# class AE_standard(nn.Module):
#     def __init__(self, input_size: int, hidden_size: int):
#         super(AE_standard, self).__init__()
#
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size * 4),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 4, hidden_size * 2),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 2, hidden_size)
#         )
#
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 2, hidden_size * 4),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 4, input_size)
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
#     def compress(self, x):
#         x = self.encoder(x)
#         return x


# Define the Denoising Autoencoder architecture for weight compression.
class Compressor_w(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Compressor_w, self).__init__()
        self.compressor = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        x = self.compressor(x)
        return x


# # Define the Denoising Autoencoder architecture for weight compression.
# class DenoisingAEw(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(DenoisingAEw, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size * 2),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 2, hidden_size),
#             nn.Softmax(dim=1)
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 2),
#             nn.ReLU(True),
#             nn.Linear(hidden_size * 2, input_size)
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
#
#     def compress(self, x):
#         x = self.encoder(x)
#         return x
#
#
#
# #### Reward model
# class RewardModel(nn.Module):
#     def __init__(self, input_size, hidden_sizes, output_size):
#         # e.g., hidden_sizes = [20, 15]  # Number of neurons in each hidden layer. Can be []
#         super(RewardModel, self).__init__()
#         self.input_size = input_size
#         self.hidden_sizes = hidden_sizes
#         self.output_size = output_size
#
#         layers = []
#         layer_sizes = [input_size] + hidden_sizes + [output_size]
#         for i in range(len(layer_sizes) - 1):
#             layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
#
#         self.layers = nn.ModuleList(layers)
#
#     def forward(self, x):
#         for layer in self.layers[:-1]:
#             x = F.leaky_relu(layer(x))
#         x = th.sigmoid(self.layers[-1](x))
#         return x
#
#     def train(self, input, label, optimizer):
#         optimizer.zero_grad()  # Clear gradients
#
#         output = self.forward(input)
#         product = output * label  # Elementwise product
#         loss = criterion(product, th.ones_like(product))  # Binary cross-entropy loss
#
#         loss.backward()  # Backpropagation
#         optimizer.step()  # Update parameters








#
# # Hyperparameters
# input_size = 16
# hidden_size = 4
# num_epochs = 20
# batch_size = 64
# learning_rate = 1e-3
#
# # Create a toy dataset with input size 16
# # Here we'll generate some random data for demonstration purposes
# data = torch.randn(1000, input_size)  # 1000 samples of size 16
# dataset = TensorDataset(data, data)  # Use the same data as input and target
# train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#
# # Model, loss function, optimizer
# model = Autoencoder(input_size=input_size, hidden_size=hidden_size).cuda()  # Use .cuda() if you have a GPU
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
# # Training loop
# for epoch in range(num_epochs):
#     for inputs, targets in train_loader:
#         inputs, targets = inputs.cuda(), targets.cuda()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#
# # Save the trained model
# torch.save(model.state_dict(), 'autoencoder.pth')
