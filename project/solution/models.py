"""Hold all models you wish to train."""
import torch
import torch.nn.functional as F

from torch import nn

from xcpetion import build_xception_backbone


class SimpleNet(nn.Module):
    """Simple Convolutional and Fully Connect network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=(7, 7))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(7, 7))
        self.conv3 = nn.Conv2d(16, 24, kernel_size=(7, 7))
        self.fc1 = nn.Linear(24 * 26 * 26, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, image):
        """Compute a forward pass."""
        first_conv_features = self.pool(F.relu(self.conv1(image)))
        second_conv_features = self.pool(F.relu(self.conv2(
            first_conv_features)))
        third_conv_features = self.pool(F.relu(self.conv3(
            second_conv_features)))
        # flatten all dimensions except batch
        flattened_features = torch.flatten(third_conv_features, 1)
        fully_connected_first_out = F.relu(self.fc1(flattened_features))
        fully_connected_second_out = F.relu(self.fc2(fully_connected_first_out))
        two_way_output = self.fc3(fully_connected_second_out)
        return two_way_output


class CustomNetwork(nn.Module):
    def __init__(self, original_model):
        super(CustomNetwork, self).__init__()

        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.mlp = nn.Sequential(
                    nn.Linear(2048, 1000),
                    nn.ReLU(),
                    nn.Linear(1000, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)
                    )

        # for p in self.features.parameters():
        #     p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        y = self.mlp(f)
        return y

def get_xception_based_model() -> nn.Module:
    """Return an Xception-Based network.

    (1) Build an Xception pre-trained backbone and hold it as `custom_network`.
    (2) Override `custom_network`'s fc attribute with the binary
    classification head stated in the exercise.
    """
    """INSERT YOUR CODE HERE, overrun return."""

    custom_network = build_xception_backbone(pretrained=True)
    # print(get_nof_params(custom_network))

    custom_network = CustomNetwork(custom_network)

    # print(get_nof_params(custom_network))

    return custom_network
    # return SimpleNet()


# def get_nof_params(model: nn.Module) -> int:
#     """Return the number of trainable model parameters.
#
#     Args:
#         model: nn.Module.
#
#     Returns:
#         The number of model parameters.
#     """
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
#
#
# get_xception_based_model()

