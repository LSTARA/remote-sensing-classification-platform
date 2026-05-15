# import torch
# import torch.nn as nn
# from torchvision import transforms
# import torch.nn.functional as F
# from PIL import Image
# import time
# import os
# import numpy as np
# import sys
#
# # 添加 cgfnet_models 到路径
# CGFNET_PATH = os.path.join(os.path.dirname(__file__), 'cgfnet_models')
# if CGFNET_PATH not in sys.path:
#     sys.path.insert(0, CGFNET_PATH)
#
# print(f"CGFNet 路径: {CGFNET_PATH}")
#
# # 直接导入
# try:
#     from main_models import ResNet50 as CGFNetResNet50
#     print("✓ CGFNet 模型导入成功")
#     USE_CGFNET = True
# except Exception as e:
#     print(f"✗ 导入失败: {e}")
#     USE_CGFNET = False
#
#
# class RemoteSensingClassifier:
#     """CGFNet 模型分类器"""
#
#     def __init__(self, model_path, att_type='ours', num_classes=45):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         print(f"使用设备: {self.device}")
#         print(f"模型类型: CGFNet (att_type={att_type})")
#
#         self.att_type = att_type
#         self.num_classes = num_classes
#         self.model = self.load_model(model_path)
#         self.transform = self.get_transform()
#         self.class_names = self.get_class_names()
#
#     def get_class_names(self):
#         return [
#             'airplane', 'airport', 'baseball_diamond', 'basketball_court', 'beach',
#             'bridge', 'chaparral', 'church', 'circular_farmland', 'cloud',
#             'commercial_area', 'dense_residential', 'desert', 'forest', 'freeway',
#             'golf_course', 'ground_track_field', 'harbor', 'industrial_area',
#             'intersection', 'island', 'lake', 'meadow', 'medium_residential',
#             'mobile_home_park', 'mountain', 'overpass', 'palace', 'parking_lot',
#             'railway', 'railway_station', 'rectangular_farmland', 'river',
#             'roundabout', 'runway', 'sea_ice', 'ship', 'snowberg',
#             'sparse_residential', 'stadium', 'storage_tank', 'tennis_court',
#             'terrace', 'thermal_power_station', 'wetland'
#         ]
#
#     def get_transform(self):
#         return transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#         ])
#
#     def load_model(self, model_path):
#         try:
#             if not USE_CGFNET:
#                 raise ImportError("CGFNet 模型未导入")
#
#             print("创建 CGFNet 模型...")
#             model = CGFNetResNet50(
#                 num_classes=self.num_classes,
#                 att_type=self.att_type,
#                 reduction_ratio=16,
#                 beta=4.0,
#                 method_version='original',
#                 channel_groups=64,
#                 sa_kernel_size=3,
#                 no_spatial=False,
#                 no_lofct=False
#             )
#
#             print(f"加载权重: {model_path}")
#             checkpoint = torch.load(model_path, map_location=self.device)
#
#             if 'state_dict' in checkpoint:
#                 state_dict = checkpoint['state_dict']
#             elif 'model' in checkpoint:
#                 state_dict = checkpoint['model']
#             else:
#                 state_dict = checkpoint
#
#             new_state_dict = {}
#             for k, v in state_dict.items():
#                 name = k.replace('module.', '')
#                 name = name.replace('_orig_mod.', '')
#                 new_state_dict[name] = v
#
#             model.load_state_dict(new_state_dict, strict=False)
#             model = model.to(self.device)
#             model.eval()
#
#             if 'epoch' in checkpoint:
#                 print(f"✓ 训练轮次: {checkpoint['epoch']}")
#             if 'best_val_ac_score' in checkpoint:
#                 print(f"✓ 最佳验证准确率: {checkpoint['best_val_ac_score']:.2f}%")
#
#             print("✓ CGFNet 模型加载成功")
#             return model
#
#         except Exception as e:
#             print(f"模型加载错误: {e}")
#             import traceback
#             traceback.print_exc()
#             raise
#
#     def predict(self, image_path):
#         start_time = time.time()
#
#         try:
#             image = Image.open(image_path).convert('RGB')
#             input_tensor = self.transform(image).unsqueeze(0).to(self.device)
#
#             with torch.no_grad():
#                 outputs = self.model(input_tensor)
#                 if isinstance(outputs, tuple):
#                     outputs = outputs[0]
#
#                 probabilities = F.softmax(outputs, dim=1)
#                 probabilities_np = probabilities.cpu().numpy()[0]
#
#             confidence = float(np.max(probabilities_np))
#             predicted = int(np.argmax(probabilities_np))
#             predicted_class = self.class_names[predicted]
#
#             top5_indices = np.argsort(probabilities_np)[-5:][::-1]
#             top5_predictions = []
#             for idx in top5_indices:
#                 top5_predictions.append({
#                     'class': self.class_names[idx],
#                     'confidence': float(probabilities_np[idx] * 100)
#                 })
#
#             processing_time = time.time() - start_time
#
#             return {
#                 'predicted_class': predicted_class,
#                 'confidence': confidence * 100,
#                 'processing_time': processing_time,
#                 'top5_predictions': top5_predictions
#             }
#
#         except Exception as e:
#             print(f"预测错误: {e}")
#             raise
#
#
# classifier = None
#
#
# def get_classifier():
#     global classifier
#     if classifier is None:
#         model_paths = [
#             'models/ours_resnet50_NWPU45_02_model_best.pth.tar',
#             'ours_resnet50_NWPU45_02_model_best.pth.tar',
#         ]
#
#         model_path = None
#         for path in model_paths:
#             if os.path.exists(path):
#                 model_path = path
#                 break
#
#         if model_path is None:
#             raise FileNotFoundError("找不到模型文件！")
#
#         print(f"使用模型: {model_path}")
#         classifier = RemoteSensingClassifier(model_path, att_type='ours', num_classes=45)
#
#     return classifier

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import time
import os
import numpy as np
import sys
import timm

# 添加 cgfnet_models 到路径
CGFNET_PATH = os.path.join(os.path.dirname(__file__), 'cgfnet_models')
if CGFNET_PATH not in sys.path:
    sys.path.insert(0, CGFNET_PATH)

print(f"CGFNet 路径: {CGFNET_PATH}")

# 导入 CGFNet
try:
    from main_models import ResNet50 as CGFNetResNet50

    print("✓ CGFNet 模型导入成功")
    USE_CGFNET = True
except Exception as e:
    print(f"✗ CGFNet 导入失败: {e}")
    USE_CGFNET = False


class CGFNetClassifier:
    """CGFNet 模型分类器（高精度）"""

    def __init__(self, model_path, att_type='ours', num_classes=45):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"模型类型: CGFNet (att_type={att_type})")

        self.att_type = att_type
        self.num_classes = num_classes
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.model_name = "CGFNet"

    def get_class_names(self):
        return [
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
        ]

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        try:
            if not USE_CGFNET:
                raise ImportError("CGFNet 模型未导入")

            print("创建 CGFNet 模型...")
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

            print(f"加载权重: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('module.', '')
                name = name.replace('_orig_mod.', '')
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict, strict=False)
            model = model.to(self.device)
            model.eval()

            if 'epoch' in checkpoint:
                print(f"✓ 训练轮次: {checkpoint['epoch']}")
            if 'best_val_ac_score' in checkpoint:
                print(f"✓ 最佳验证准确率: {checkpoint['best_val_ac_score']:.2f}%")

            print("✓ CGFNet 模型加载成功")
            return model

        except Exception as e:
            print(f"CGFNet 模型加载错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, image_path):
        start_time = time.time()

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                probabilities = F.softmax(outputs, dim=1)
                probabilities_np = probabilities.cpu().numpy()[0]

            confidence = float(np.max(probabilities_np))
            predicted = int(np.argmax(probabilities_np))
            predicted_class = self.class_names[predicted]

            top5_indices = np.argsort(probabilities_np)[-5:][::-1]
            top5_predictions = []
            for idx in top5_indices:
                top5_predictions.append({
                    'class': self.class_names[idx],
                    'confidence': float(probabilities_np[idx] * 100)
                })

            processing_time = time.time() - start_time

            return {
                'predicted_class': predicted_class,
                'confidence': confidence * 100,
                'processing_time': processing_time,
                'top5_predictions': top5_predictions,
                'model_name': self.model_name
            }

        except Exception as e:
            print(f"预测错误: {e}")
            raise


class EfficientNetClassifier:
    """EfficientNet 模型分类器（快速响应）"""

    def __init__(self, model_name='efficientnet_b3', num_classes=45):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        print(f"模型类型: EfficientNet ({model_name})")

        self.model_name = model_name
        self.num_classes = num_classes
        self.model = self.load_model()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.display_name = "EfficientNet-B3"

    def get_class_names(self):
        return [
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
        ]

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            print(f"加载 EfficientNet 模型: {self.model_name}")

            model = timm.create_model(self.model_name, pretrained=True)

            # 替换分类头
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_features, self.num_classes)
                )
            elif hasattr(model, 'fc'):
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, self.num_classes)

            model = model.to(self.device)
            model.eval()

            print(f"✓ EfficientNet 模型加载成功")
            return model

        except Exception as e:
            print(f"EfficientNet 模型加载错误: {e}")
            import traceback
            traceback.print_exc()
            raise

    def predict(self, image_path):
        start_time = time.time()

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                probabilities_np = probabilities.cpu().numpy()[0]

            confidence = float(np.max(probabilities_np))
            predicted = int(np.argmax(probabilities_np))
            predicted_class = self.class_names[predicted]

            top5_indices = np.argsort(probabilities_np)[-5:][::-1]
            top5_predictions = []
            for idx in top5_indices:
                top5_predictions.append({
                    'class': self.class_names[idx],
                    'confidence': float(probabilities_np[idx] * 100)
                })

            processing_time = time.time() - start_time

            return {
                'predicted_class': predicted_class,
                'confidence': confidence * 100,
                'processing_time': processing_time,
                'top5_predictions': top5_predictions,
                'model_name': self.display_name
            }

        except Exception as e:
            print(f"预测错误: {e}")
            raise


# 全局变量
cgfnet_classifier = None
efficientnet_classifier = None


def get_classifier(model_choice='cgfnet'):
    """获取分类器实例

    Args:
        model_choice: 模型选择，'cgfnet' 或 'efficientnet'
    """
    global cgfnet_classifier, efficientnet_classifier

    if model_choice == 'efficientnet':
        if efficientnet_classifier is None:
            print("初始化 EfficientNet 模型...")
            efficientnet_classifier = EfficientNetClassifier(model_name='efficientnet_b3', num_classes=45)
        return efficientnet_classifier
    else:
        if cgfnet_classifier is None:
            print("初始化 CGFNet 模型...")
            model_paths = [
                'models/ours_resnet50_NWPU45_02_model_best.pth.tar',
                'ours_resnet50_NWPU45_02_model_best.pth.tar',
            ]
            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError("找不到 CGFNet 模型文件！")
            print(f"使用模型: {model_path}")
            cgfnet_classifier = CGFNetClassifier(model_path, att_type='ours', num_classes=45)
        return cgfnet_classifier