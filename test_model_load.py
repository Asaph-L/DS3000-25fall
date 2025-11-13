"""Test script to verify model loading backward compatibility"""
import torch
import torchvision
import torchvision.models as models
from pathlib import Path
import os

def build_resnet50_classifier(num_classes=23):
    """Build ResNet50 model architecture"""
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

def build_efficientnet_classifier(num_classes=23):
    """Build EfficientNet-B0 model architecture"""
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def load_model(model_path, device='cpu'):
    """
    Load a trained PyTorch model with backward compatibility.

    Supports two formats:
    1. Complete checkpoint (with metadata: model_state_dict, history, class_to_idx, etc.)
    2. Legacy checkpoint (weights only: state_dict)

    Args:
        model_path: Path to the saved model checkpoint
        device: Device to load the model on ('cpu' or 'cuda')

    Returns:
        tuple: (model, class_to_idx, idx_to_class)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a complete checkpoint or just weights
    is_complete = isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint

    if is_complete:
        # ========== Complete checkpoint format ==========
        print("→ Detected: Complete checkpoint (with metadata)")

        # Extract metadata
        num_classes = checkpoint['num_classes']
        model_name = checkpoint['model_name']
        class_to_idx = checkpoint['class_to_idx']

        # Build model architecture
        if 'resnet' in model_name.lower():
            model = build_resnet50_classifier(num_classes)
        elif 'efficientnet' in model_name.lower():
            model = build_efficientnet_classifier(num_classes)
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        # Create idx_to_class mapping
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        print(f"✓ Loaded {model_name} with {num_classes} classes")

    else:
        # ========== Legacy checkpoint format (weights only) ==========
        print("→ Detected: Legacy checkpoint (weights only)")

        # Infer num_classes from the weights
        if 'fc.weight' in checkpoint:  # ResNet50
            num_classes = checkpoint['fc.weight'].shape[0]
            model_type = 'resnet50'
            print(f"→ Detected model type: ResNet50")
        elif 'classifier.1.weight' in checkpoint:  # EfficientNet
            num_classes = checkpoint['classifier.1.weight'].shape[0]
            model_type = 'efficientnet'
            print(f"→ Detected model type: EfficientNet-B0")
        else:
            raise ValueError("Cannot determine model type from checkpoint keys")

        # Build model architecture
        if model_type == 'resnet50':
            model = build_resnet50_classifier(num_classes)
        else:
            model = build_efficientnet_classifier(num_classes)

        # Load weights
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        # Reconstruct class mapping from CLEAN_DIR
        print(f"→ Reconstructing class mappings from dataset...")

        # Try to find CLEAN_DIR
        ROOT = Path().resolve()
        CLEAN_DIR = os.path.join(ROOT, "data", "clean")

        if os.path.exists(CLEAN_DIR):
            clean_dataset = torchvision.datasets.ImageFolder(root=CLEAN_DIR)
            class_to_idx = clean_dataset.class_to_idx
            idx_to_class = {v: k for k, v in class_to_idx.items()}
            print(f"✓ Reconstructed class mappings from {CLEAN_DIR}")
        else:
            print(f"⚠ Warning: {CLEAN_DIR} not found")
            print(f"→ Creating placeholder class mappings (0-{num_classes-1})")
            class_to_idx = {f"class_{i}": i for i in range(num_classes)}
            idx_to_class = {i: f"class_{i}" for i in range(num_classes)}

        print(f"✓ Loaded legacy checkpoint with {num_classes} classes")

    return model, class_to_idx, idx_to_class


if __name__ == "__main__":
    print("="*70)
    print("Testing Model Loading Backward Compatibility")
    print("="*70)

    device = torch.device("cpu")
    ROOT = Path().resolve()

    # Test 1: Load legacy ResNet50
    print("\n[Test 1] Loading legacy ResNet50...")
    try:
        resnet_path = os.path.join(ROOT, "best_resnet50.pth")
        resnet_model, resnet_class_to_idx, resnet_idx_to_class = load_model(resnet_path, device)
        print(f"  ✓ ResNet50 loaded successfully")
        print(f"  → Number of classes: {len(resnet_idx_to_class)}")
        print(f"  → Sample classes: {list(resnet_idx_to_class.values())[:3]}")
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 2: Load legacy EfficientNet
    print("\n[Test 2] Loading legacy EfficientNet-B0...")
    try:
        efficientnet_path = os.path.join(ROOT, "best_efficientnet_b0.pth")
        efficientnet_model, efficientnet_class_to_idx, efficientnet_idx_to_class = load_model(efficientnet_path, device)
        print(f"  ✓ EfficientNet-B0 loaded successfully")
        print(f"  → Number of classes: {len(efficientnet_idx_to_class)}")
        print(f"  → Sample classes: {list(efficientnet_idx_to_class.values())[:3]}")
    except FileNotFoundError as e:
        print(f"  ⚠ {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    # Test 3: Load complete checkpoint (if exists)
    print("\n[Test 3] Loading complete checkpoint (if exists)...")
    try:
        complete_path = os.path.join(ROOT, "resnet50_complete.pth")
        complete_model, complete_class_to_idx, complete_idx_to_class = load_model(complete_path, device)
        print(f"  ✓ Complete checkpoint loaded successfully")
        print(f"  → Number of classes: {len(complete_idx_to_class)}")
    except FileNotFoundError as e:
        print(f"  ⚠ {e} (This is expected if you haven't saved complete checkpoints yet)")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print("\n" + "="*70)
    print("Testing Complete!")
    print("="*70)

