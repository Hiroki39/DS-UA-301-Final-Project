import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.functional as F
from extras.encoder import ResnetEncoder


def collate_fn(batch):
    return tuple(zip(*batch))


class ManipDetectionModel(nn.Module):

    def __init__(self, base=18, pretrained=False, freeze_base=False):
        super(ManipDetectionModel, self).__init__()

        # for each grid in the feature map we have 3 anchors of sizes: 40x40, 50x50, 60x60
        num_anchors = 70

        # regular resnet 18 encoder
        self.encoder = ResnetEncoder(num_layers=base, pretrained=pretrained)

        if pretrained and freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False

        channel = 256 if base < 50 else 1024
        # a small conv net
        self.conv = nn.Conv2d(
            channel, channel, kernel_size=3, stride=1, padding=1
        )

        # Add a Convolutional Layer to prediction the class predictions. This is a head that predicts whether a chunk/anchor contains an object or not.
        self.cls_logits = nn.Conv2d(
            channel, num_anchors * 3, kernel_size=1, padding=0)

        # Add a Convolutional Layer to prediction the class predictions. This is a head that regresses over the 4 bounding box offsets for each anchor
        self.bbox_pred = nn.Conv2d(
            channel, num_anchors * 4, kernel_size=1, padding=0)

    def permute_and_flatten(self, layer, N, A, C, H, W):
        # helper function that rearranges the input for the loss function
        layer = layer.view(N, -1, C, H, W)
        layer = layer.permute(0, 3, 4, 1, 2)
        layer = layer.reshape(N, -1, C)
        return layer

    def get_predict_regressions(self, cls_pred, box_pred):
        # helper function that gets outputs in the right shape for applying the loss
        N, AxC, H, W = cls_pred.shape
        Ax4 = box_pred.shape[1]
        A = Ax4 // 4
        C = AxC // A
        cls_pred = self.permute_and_flatten(
            cls_pred, N, A, C, H, W
        )

        box_pred = self.permute_and_flatten(
            box_pred, N, A, 4, H, W
        )
        return cls_pred, box_pred

    def forward(self, x):
        bt_sz = x.size(0)

        # we take the 3rd output feature map of size 8 x 8 from
        # the resnet18 encoder this means that the stride
        # is 16 as our input image is 128x128 in size.
        x = self.encoder(x)[3]

        x = F.relu(self.conv(x))

        cls_pred = self.cls_logits(x)
        box_pred = self.bbox_pred(x)

        cls_pred, box_pred = self.get_predict_regressions(cls_pred, box_pred)
        cls_pred = cls_pred.permute(0, 2, 1)

        return cls_pred.squeeze(2), box_pred


def draw_box(ax, box):
    color = 'black'
    ex1 = box[0]
    ey1 = box[1]
    ex2 = box[2]
    ey2 = box[3]
    rect = patches.Rectangle((ex1, ey1), abs(
        ex1 - ex2), abs(ey1 - ey2), linewidth=2, edgecolor=color, fill=False)
    ax.add_patch(rect)
