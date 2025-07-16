from torch.nn import Module, Sequential, Conv1d, BatchNorm1d, MaxPool1d, AdaptiveAvgPool1d, Flatten, Linear, Dropout, ReLU

class DynamicCNN(Module):
    def __init__(self, config):
        super().__init__()

        layers = []
        for layer in config:
            match layer["type"].lower():
                case "conv":
                    layers.append(Conv1d(
                        in_channels=layer["in_channels"],
                        out_channels=layer["out_channels"],
                        kernel_size=layer["kernel_size"],
                        stride=layer.get("stride", 1),
                        padding=layer.get("padding", 0)
                    ))
                case "batchnorm": layers.append(BatchNorm1d(num_features=layer["num_features"]))
                case "maxpool": layers.append(MaxPool1d(kernel_size=layer["kernel_size"], stride=layer["stride"]))
                case "adaptiveavgpool": layers.append(AdaptiveAvgPool1d(output_size=layer["output_size"]))
                case "flatten": layers.append(Flatten())
                case "linear": layers.append(Linear(in_features=layer["in_features"], out_features=layer["out_features"]))
                case "dropout": layers.append(Dropout(p=layer["p"]))
                case "relu": layers.append(ReLU())
                case "_": raise ValueError(f"Unsupported layer type: {layer['type']}")
        self.model = Sequential(*layers)

    def forward(self, x): return self.model(x)