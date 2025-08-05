from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, MaxPool1d, AdaptiveAvgPool1d, Flatten, Linear, Dropout, ReLU, GELU

LAYER_REGISTRY = {
    "conv1d": lambda l: Conv1d(
        in_channels=l["in_channels"],
        out_channels=l["out_channels"],
        kernel_size=l["kernel_size"],
        stride=l["stride"],
        padding=l["padding"],
        bias=l["bias"]
    ),
    "batchnorm1d": lambda l: BatchNorm1d(num_features=l["num_features"]),
    "maxpool1d": lambda l: MaxPool1d(kernel_size=l["kernel_size"], stride=l["stride"]),
    "adaptiveavgpool1d": lambda l: AdaptiveAvgPool1d(output_size=l["output_size"]),
    "flatten": lambda l: Flatten(),
    "linear": lambda l: Linear(in_features=l["in_features"], out_features=l["out_features"]),
    "dropout": lambda l: Dropout(p=l["p"]),
    "relu": lambda l: ReLU(),
    "gelu": lambda l: GELU()
}

class DynamicCNN(Module):
    def __init__(self, architecture):
        super().__init__()
        layers = []
        for layer in architecture:
            layer_type = layer["type"].lower()
            if layer_type not in LAYER_REGISTRY: raise ValueError(f"Unsupported layer type: {layer['type']}")
            layers.append(LAYER_REGISTRY[layer_type](layer))
        self.model = Sequential(*layers)

    def forward(self, x): return self.model(x)