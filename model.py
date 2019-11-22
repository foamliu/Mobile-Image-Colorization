import torch
from torchscope import scope

from config import num_classes
from models.deeplab import DeepLab

if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16, num_classes=num_classes)
    model.eval()
    input = torch.rand(1, 1, 256, 256)
    output = model(input)
    print(output.size())
    scope(model, (1, 256, 256))

    # model = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=num_classes)
    # model.eval()
    # input = torch.rand(1, 3, 256, 256)
    # output = model(input)['out']
    # print(output.size())
    # scope(model, (3, 256, 256))
