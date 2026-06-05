import torch
import torch.nn as nn
import timm


class DeepFakeHybridModel(nn.Module):

    def __init__(self):

        super().__init__()

        # ======================================
        # CNN Branch
        # ======================================

        self.cnn_branch = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0
        )

        self.cnn_features = (
            self.cnn_branch.num_features
        )  # 1280

        # ======================================
        # ViT Branch
        # ======================================

        self.vit_branch = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0
        )

        self.vit_features = (
            self.vit_branch.num_features
        )  # 384

        # ======================================
        # Feature Fusion
        # ======================================

        fusion_dim = (
            self.cnn_features +
            self.vit_features
        )

        # ======================================
        # Classification Head
        # ======================================

        self.classifier = nn.Sequential(

            nn.Linear(
                fusion_dim,
                512
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                512
            ),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                2
            )
        )

    def forward(self, x):

        # ==================================
        # CNN Features
        # ==================================

        cnn_features = self.cnn_branch(x)

        # ==================================
        # ViT Features
        # ==================================

        vit_features = self.vit_branch(x)

        # ==================================
        # Feature Fusion
        # ==================================

        fused_features = torch.cat(
            (
                cnn_features,
                vit_features
            ),
            dim=1
        )

        # ==================================
        # Classification
        # ==================================

        output = self.classifier(
            fused_features
        )

        return output