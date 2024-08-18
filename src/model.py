import torch
from torch import nn


class NumRecCNN(nn.Module):
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, out_features=output_shape)
    )

  def forward(self, x: torch.Tensor):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.classifier(x)
    return x

  def save(self, MODEL_PATH = Path("models"), MODEL_NAME = "num_rec_cnn.pth"):
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    #print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=self.state_dict(), f=MODEL_SAVE_PATH)

  def load(self, MODEL_PATH = Path("models"), MODEL_NAME = "num_rec_cnn.pth"):
    self.load_state_dict(torch.load(f = MODEL_PATH / MODEL_NAME))
    self.eval()