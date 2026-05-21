import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=False)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:],
                            mode='bilinear', align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def apply_heatmap(original_image, cam, alpha=0.5):
    """将热力图叠加到原图上"""
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image.convert('RGB'))

    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap + (1 - alpha) * original_image)
    return Image.fromarray(overlay)


def find_target_layer(model, model_type='cgfnet'):
    """自动查找用于 Grad-CAM 的目标卷积层"""
    if model_type == 'cgfnet':
        # ResNet50: layer4 最后一个 Bottleneck 的 conv3
        return model.layer4[-1].conv3
    elif model_type == 'lwganet':
        # LWGANet: 遍历 stages 找到最后一个 Conv2d
        last_conv = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv
    return None
