import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib

IMG_SIZE = 224
BOX = (0.20, 0.90, 0.15, 0.85)          # same chest box used in training
CLASSES = ["NORMAL", "PNEUMONIA"]


class ChestCrop:
    def __call__(self, img):
        w, h = img.size
        t, b, l, r = BOX
        return img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


preprocess = transforms.Compose([
    ChestCrop(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(path):
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v
                           for k, v in state.items()})   # float16 file -> float32 model
    model.eval()
    return model


def looks_like_xray(img):
    hsv = np.asarray(img.convert("RGB").convert("HSV")).astype(np.float32) / 255
    return hsv[:, :, 1].mean() < 0.10      # X-rays have almost no colour saturation


def predict_with_gradcam(model, img):
    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0)
    m = model

    with torch.enable_grad():
        # Run ResNet-18 step by step so we can keep the last feature maps
        feats = m.layer4(m.layer3(m.layer2(m.layer1(
                    m.maxpool(m.relu(m.bn1(m.conv1(x))))))))
        output = m.fc(torch.flatten(m.avgpool(feats), 1))
        probs = torch.softmax(output, dim=1)[0].tolist()
        pred = int(output.argmax(1).item())

        # Grad-CAM: how much each feature map mattered for the predicted class
        grads = torch.autograd.grad(output[0, pred], feats)[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear")
    cam = (cam / (cam.max() + 1e-8)).squeeze().detach().numpy()

    crop = ChestCrop()(img).resize((IMG_SIZE, IMG_SIZE))
    heat = matplotlib.colormaps["jet"](cam)[:, :, :3]
    blended = 0.6 * np.asarray(crop, dtype=np.float32) / 255 + 0.4 * heat
    overlay = Image.fromarray((blended * 255).astype(np.uint8)).resize((448, 448))

    return {CLASSES[0]: probs[0], CLASSES[1]: probs[1]}, overlay
