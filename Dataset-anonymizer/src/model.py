# model.py

import torch.nn as nn
import torchvision.models as models
from opacus.validators import ModuleValidator


class ModelBuilder:
    @staticmethod
    # Build a custom model specifically adapted to MNIST dataset
    def build_model():
        # Load ResNet18 configured for 10 output classes
        model = models.resnet18(num_classes=10)

        # Modify the first convolutional layer to accept single-channel images
        # ResNet18 is originally designed to accept three-channel images (red, green and blue) (i.e., color images)
        # We work with grayscale images (one color) --> adapt ResNet18
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 10)

        return ModelBuilder.validate_model(model)

    # Validates the model using opacus’s ModuleValidator to ensure
    # if it is compatible with the private differential training framework
    @staticmethod
    def validate_model(model):
        errors = ModuleValidator.validate(model, strict=False)
        if errors:
            model = ModuleValidator.fix(model)
            errors = ModuleValidator.validate(model, strict=False)
            assert not errors, "The model is not compliant after correction"
        return model
