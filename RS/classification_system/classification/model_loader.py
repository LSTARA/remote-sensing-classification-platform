import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import time
import os
import numpy as np
import sys

# 添加 cgfnet_models 到路径
CGFNET_PATH = os.path.join(os.path.dirname(__file__), 'cgfnet_models')
if CGFNET_PATH not in sys.path:
    sys.path.insert(0, CGFNET_PATH)

print(f"CGFNet path: {CGFNET_PATH}")

# 导入 CGFNet
try:
    from main_models import ResNet50 as CGFNetResNet50
    print("[OK] CGFNet model imported")
    USE_CGFNET = True
except Exception as e:
    print(f"[FAIL] CGFNet import failed: {e}")
    USE_CGFNET = False

# 导入 LWGANet
try:
    from .lwganet_model import LWGANet_L0_1242_e32_k11_GELU as LWGANet_L0
    print("[OK] LWGANet model imported")
    USE_LWGANET = True
except Exception as e:
    print(f"[FAIL] LWGANet import failed: {e}")
    USE_LWGANET = False


# ==================== 数据集配置 ====================

DATASET_CONFIG = {
    'AID': {
        'num_classes': 30,
        'class_names': [
            'airport', 'bare_land', 'baseball_field', 'beach', 'bridge',
            'center', 'church', 'commercial', 'dense_residential', 'desert',
            'farmland', 'forest', 'industrial', 'meadow', 'medium_residential',
            'mountain', 'park', 'parking', 'playground', 'pond',
            'port', 'railway_station', 'resort', 'river', 'school',
            'sparse_residential', 'square', 'stadium', 'storage_tanks', 'viaduct'
        ],
        'class_names_cn': [
            '机场', '裸地', '棒球场', '海滩', '桥梁',
            '中心区', '教堂', '商业区', '密集住宅区', '沙漠',
            '农田', '森林', '工业区', '草地', '中等住宅区',
            '山脉', '公园', '停车场', '操场', '池塘',
            '港口', '火车站', '度假区', '河流', '学校',
            '稀疏住宅区', '广场', '体育场', '储油罐', '高架桥'
        ],
        'label': 'AID (30 classes)',
    },
    'NWPU': {
        'num_classes': 45,
        'class_names': [
            'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
            'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
            'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
            'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
            'intersection', 'island', 'lake', 'meadow', 'medium_residential',
            'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
            'railway', 'railway_station', 'rectangular_farmland', 'river',
            'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg',
            'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
            'terrace', 'thermal_power_station', 'wetland'
        ],
        'class_names_cn': [
            '飞机', '机场', '棒球场', '篮球场', '海滩',
            '桥梁', '灌木丛', '教堂', '圆形农田', '云',
            '商业区', '密集住宅区', '沙漠', '森林', '高速公路',
            '高尔夫球场', '田径场', '港口', '工业区',
            '交叉路口', '岛屿', '湖泊', '草地', '中等住宅区',
            '移动房车公园', '山脉', '立交桥', '宫殿', '停车场',
            '铁路', '火车站', '矩形农田', '河流',
            '环岛', '跑道', '海冰', '船只', '雪山',
            '稀疏住宅区', '体育场', '储油罐', '网球场',
            '梯田', '热电站', '湿地'
        ],
        'label': 'NWPU-RESISC45 (45 classes)',
    },
    'UCM': {
        'num_classes': 21,
        'class_names': [
            'agricultural', 'airplane', 'baseball_diamond', 'beach', 'buildings',
            'chaparral', 'dense_residential', 'forest', 'freeway', 'golf_course',
            'harbor', 'intersection', 'medium_residential', 'mobile_home_park',
            'overpass', 'parking_lot', 'river', 'runway', 'sparse_residential',
            'storage_tanks', 'tennis_court'
        ],
        'class_names_cn': [
            '农田', '飞机', '棒球场', '海滩', '建筑',
            '灌木丛', '密集住宅区', '森林', '高速公路', '高尔夫球场',
            '港口', '交叉路口', '中等住宅区', '移动房车公园',
            '立交桥', '停车场', '河流', '跑道', '稀疏住宅区',
            '储油罐', '网球场'
        ],
        'label': 'UC Merced (21 classes)',
    },
}

# (model, dataset) -> weight file path
MODEL_WEIGHT_PATHS = {
    ('cgfnet', 'AID'):  'models/ours_resnet50_AID02_model_best.pth.tar',
    ('cgfnet', 'NWPU'): 'models/ours_resnet50_NWPU45_02_model_best.pth.tar',
    ('cgfnet', 'UCM'):  'models/ours_resnet50_UCM08_model_best.pth.tar',
    ('lwganet', 'AID'):  'models/best-val_acc_AID.pth',
    ('lwganet', 'NWPU'): 'models/best-val_acc_NWPU.pth',
    ('lwganet', 'UCM'):  'models/best-val_acc_UCM.pth',
}


# (model, dataset) -> validation accuracy (%)
MODEL_ACCURACY = {
    ('cgfnet', 'AID'):  96.85,
    ('cgfnet', 'NWPU'): 96.02,
    ('cgfnet', 'UCM'):  99.29,
    ('lwganet', 'AID'):  88.31,
    ('lwganet', 'NWPU'): 88.66,
    ('lwganet', 'UCM'):  91.00,
}


# ==================== 分类器类 ====================

class CGFNetClassifier:
    """CGFNet 分类器"""

    def __init__(self, model_path, dataset_code, att_type='ours'):
        cfg = DATASET_CONFIG[dataset_code]
        self.dataset_code = dataset_code
        self.num_classes = cfg['num_classes']
        self.class_names = cfg['class_names']
        self.class_names_cn = cfg.get('class_names_cn', cfg['class_names'])
        self.display_name = f"CGFNet + {cfg['label']}"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        print(f"Model: {self.display_name}")

        self.att_type = att_type
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not USE_CGFNET:
            raise ImportError("CGFNet model not imported")

        print("Creating CGFNet model...")
        model = CGFNetResNet50(
            num_classes=self.num_classes,
            att_type=self.att_type,
            reduction_ratio=16,
            beta=4.0,
            method_version='original',
            channel_groups=64,
            sa_kernel_size=3,
            no_spatial=False,
            no_lofct=False
        )

        print(f"Loading weights: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('_orig_mod.', '')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        if 'epoch' in checkpoint:
            print(f"[OK] epoch: {checkpoint['epoch']}")
        if 'best_val_ac_score' in checkpoint:
            print(f"[OK] best val acc: {checkpoint['best_val_ac_score']:.2f}%")

        print(f"[OK] {self.display_name} loaded")
        return model

    def predict(self, image_path):
        start_time = time.time()
        try:
            image = Image.open(image_path).convert('RGB')
            image = transforms.Compose([
                transforms.Resize((256, 256), Image.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            confidence = float(np.max(probs))
            predicted = int(np.argmax(probs))

            top5 = [
                {'class': self.class_names[i], 'confidence': float(probs[i] * 100)}
                for i in np.argsort(probs)[-5:][::-1]
            ]

            return {
                'predicted_class': self.class_names[predicted],
                'predicted_class_cn': self.class_names_cn[predicted] if self.class_names_cn else self.class_names[predicted],
                'confidence': confidence * 100,
                'processing_time': time.time() - start_time,
                'top5_predictions': top5,
                'model_name': self.display_name,
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

    def generate_heatmap(self, image_path, output_path=None):
        """生成 Grad-CAM 热力图"""
        from .grad_cam import GradCAM, apply_heatmap, find_target_layer
        target_layer = find_target_layer(self.model, 'cgfnet')
        if target_layer is None:
            return None

        grad_cam = GradCAM(self.model, target_layer)

        pil_img = Image.open(image_path).convert('RGB')
        tensor = transforms.Compose([
            transforms.Resize((256, 256), Image.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])(pil_img).unsqueeze(0).to(self.device)

        cam = grad_cam(tensor)
        result_img = apply_heatmap(pil_img, cam)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_img.save(output_path)

        return result_img


class LWGANetClassifier:
    """LWGANet 分类器"""

    def __init__(self, model_path, dataset_code):
        cfg = DATASET_CONFIG[dataset_code]
        self.dataset_code = dataset_code
        self.num_classes = cfg['num_classes']
        self.class_names = cfg['class_names']
        self.class_names_cn = cfg.get('class_names_cn', cfg['class_names'])
        self.display_name = f"LWGANet-L0 + {cfg['label']}"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        print(f"Model: {self.display_name}")

        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        if not USE_LWGANET:
            raise ImportError("LWGANet model not imported")

        print("Creating LWGANet-L0 model...")
        model = LWGANet_L0(num_classes=self.num_classes)

        print(f"Loading weights: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('_orig_mod.', '')
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(self.device)
        model.eval()

        if 'epoch' in checkpoint:
            print(f"[OK] epoch: {checkpoint['epoch']}")

        print(f"[OK] {self.display_name} loaded")
        return model

    def predict(self, image_path):
        start_time = time.time()
        try:
            image = Image.open(image_path).convert('RGB')
            image = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

            confidence = float(np.max(probs))
            predicted = int(np.argmax(probs))

            top5 = [
                {'class': self.class_names[i], 'confidence': float(probs[i] * 100)}
                for i in np.argsort(probs)[-5:][::-1]
            ]

            return {
                'predicted_class': self.class_names[predicted],
                'predicted_class_cn': self.class_names_cn[predicted] if self.class_names_cn else self.class_names[predicted],
                'confidence': confidence * 100,
                'processing_time': time.time() - start_time,
                'top5_predictions': top5,
                'model_name': self.display_name,
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            raise

    def generate_heatmap(self, image_path, output_path=None):
        """生成 Grad-CAM 热力图"""
        from .grad_cam import GradCAM, apply_heatmap, find_target_layer
        target_layer = find_target_layer(self.model, 'lwganet')
        if target_layer is None:
            return None

        grad_cam = GradCAM(self.model, target_layer)

        pil_img = Image.open(image_path).convert('RGB')
        tensor = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])(pil_img).unsqueeze(0).to(self.device)

        cam = grad_cam(tensor)
        result_img = apply_heatmap(pil_img, cam)

        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_img.save(output_path)

        return result_img


# ==================== 分类器缓存与获取 ====================

_classifiers = {}  # key: (model_choice, dataset_choice)


def get_classifier(model_choice='cgfnet', dataset_choice='NWPU'):
    key = (model_choice, dataset_choice)
    if key not in _classifiers:
        weight_path = MODEL_WEIGHT_PATHS.get(key)
        if weight_path is None:
            raise ValueError(f"No weight file configured for {model_choice} + {dataset_choice}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"Weight file not found: {weight_path}"
            )

        print(f"Loading: model={model_choice}, dataset={dataset_choice}, weights={weight_path}")

        if model_choice == 'lwganet':
            _classifiers[key] = LWGANetClassifier(weight_path, dataset_choice)
        else:
            _classifiers[key] = CGFNetClassifier(weight_path, dataset_choice)
    return _classifiers[key]