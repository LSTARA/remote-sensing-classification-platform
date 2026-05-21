"""诊断 CGFNet 权重加载情况"""
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'classification', 'cgfnet_models'))
from main_models import ResNet50 as CGFNetResNet50

for dataset in ['AID', 'NWPU', 'UCM']:
    weight_map = {
        'AID': 'models/ours_resnet50_AID02_model_best.pth.tar',
        'NWPU': 'models/ours_resnet50_NWPU45_02_model_best.pth.tar',
        'UCM': 'models/ours_resnet50_UCM08_model_best.pth.tar',
    }
    num_classes_map = {'AID': 30, 'NWPU': 45, 'UCM': 21}

    path = weight_map[dataset]
    nc = num_classes_map[dataset]

    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}, classes: {nc}")
    print(f"Weights: {path}")

    if not os.path.exists(path):
        print(f"  [ERROR] File not found!")
        continue

    model = CGFNetResNet50(num_classes=nc, att_type='ours', reduction_ratio=16,
                           beta=4.0, channel_groups=64, sa_kernel_size=3,
                           no_spatial=False, no_lofct=False)

    checkpoint = torch.load(path, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print(f"  Checkpoint type: state_dict, epoch={checkpoint.get('epoch', '?')}, "
              f"best_val={checkpoint.get('best_val_ac_score', '?')}")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print(f"  Checkpoint type: model")
    else:
        state_dict = checkpoint
        print(f"  Checkpoint type: raw dict")

    # Compare keys
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    matched = model_keys & ckpt_keys

    if missing:
        print(f"  Missing keys ({len(missing)}): {sorted(missing)[:5]}...")
    else:
        print(f"  Missing keys: 0")

    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:5]}...")
    else:
        print(f"  Unexpected keys: 0")

    # Check fc layer weight shape
    fc_key = 'fc.weight'
    if fc_key in state_dict:
        print(f"  fc.weight shape in checkpoint: {state_dict[fc_key].shape}")
        print(f"  fc.weight shape in model:     {model.state_dict()[fc_key].shape}")

    # Test actual loading
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace('module.', '').replace('_orig_mod.', '')] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print(f"  [WARN] After load - missing: {len(missing)} keys")
    if unexpected:
        print(f"  [WARN] After load - unexpected: {len(unexpected)} keys")

    # Quick forward test
    model.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    probs = torch.softmax(out, dim=1)[0]
    top1_val, top1_idx = probs.max(0)
    print(f"  Random input test - top1 confidence: {top1_val.item()*100:.1f}%, class idx: {top1_idx.item()}")
    print(f"  Top5: {probs.topk(5).values.tolist()}")
