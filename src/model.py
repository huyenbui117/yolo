import torch
import torch.nn as nn
architecture_config = [
    # Tuple: (kernel_size, filter, stride, padding)
    (7, 64, 2, 3),                          #[2, 64, 224, 224]
    'M',                                    #[2, 64, 112, 112]
    (3, 192, 1, 1),                         #[2, 192, 112, 112]
    'M',                                    #[2, 192, 56, 56]
    (1, 128, 1, 0),                         #[2, 128, 56, 56]
    (3, 256, 1, 1),                         #[2, 256, 56, 56]
    (1, 256, 1, 0),                         #[2, 256, 56, 56]
    (3, 512, 1, 1),                         #[2, 512, 56, 56]
    'M',                                    #[2, 512, 28, 28]
    # List: [(kernel_size, filter, stride, padding), repeat]
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    #[2, 512, 28, 28]
    (1, 512, 1, 0),                         #[2, 512, 28, 28]
    (3, 1024, 1, 1),                        #[2, 1024, 28, 28]
    'M',                                    #[2, 1024, 14, 14]
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],   #[2, 1024, 14, 14]
    (3, 1024, 1, 1),                        #[2, 1024, 14, 14]
    (3, 1024, 2, 1),                        #[2, 1024, 7, 7] 
    (3, 1024, 1, 1),                        #[2, 1024, 7, 7]
    (3, 1024, 1, 1),                        #[2, 1024, 7, 7] => flatten: [2, 50176] => fc: [2, 7*7*30]=[2, 1470]
]
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batchnorm(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.net = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.net(x)
        
        return self.fcs(torch.flatten(x, start_dim=1))
        # return torch.flatten(x, start_dim=1)
        # return x

    
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x)==tuple:
                layers+=[CNNBlock(in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3])]
                in_channels = x[1]
            elif type(x)==str:
                layers+=[nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]
            elif type(x)==list:
                conv1 = x[0]
                conv2 = x[1]
                num_repeats = x[2]
                for _ in range(num_repeats):
                    layers+=[CNNBlock(in_channels, conv1[1], kernel_size=conv1[0], stride=conv1[2], padding=conv1[3])]
                    layers+=[CNNBlock(conv1[1], conv2[1], kernel_size=conv2[0], stride=conv2[2], padding=conv2[3])]
                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 4096), # 7*7*1024, 4096 in paper
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (C + B * 5)), # 7*7*(20+2*5) ~ 7*7*30 in paper
        )
def test(S=7, B=2, C=20):
    model = YOLOv1(split_size=S, num_boxes=B, num_classes=C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape) # (N, S*S*(C + B*5)) = (N, 7*7*(20 + 2*5)) = (N, 7*7*30) = (N, 1470)

test()
