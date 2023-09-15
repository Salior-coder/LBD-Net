import torch
import torch.nn as nn
import torch.nn.functional as F

from sampling_points import sampling_points, point_sample


class PointHead(nn.Module):
    def __init__(self, in_c=135, num_classes=7, k=7, beta=0.75):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_c, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.k = k
        self.beta = beta
        # self.training = False

    def forward(self, fine, coarse):
        if not self.training:
            return self.inference(fine, coarse)

        points = sampling_points(coarse, 1024, self.k, self.beta)
        coarse = point_sample(coarse, points, align_corners=False)
        fine = point_sample(fine, points, align_corners=False)
        feature_representation = torch.cat([coarse, fine], dim=1)

        rend = self.mlp(feature_representation.transpose(1, 2)).transpose(1, 2)
        return {"rend": rend, "points": points}

    @torch.no_grad()
    def inference(self, fine, layers):
        num_points = 1024

        points_idx, points = sampling_points(layers, num_points, training=self.training)
        coarse = point_sample(layers, points, align_corners=False)
        fine = point_sample(fine, points, align_corners=False)

        feature_representation = torch.cat([coarse, fine], dim=1)
        rend = self.mlp(feature_representation.transpose(1, 2)).transpose(1, 2)

        B, C, H, W = layers.shape
        points_idx = points_idx.unsqueeze(1).expand(-1, C, -1)

        out = (layers.reshape(B, C, -1)
               .scatter_(2, points_idx, rend)
               .view(B, C, H, W))

        return {"fine": out}
