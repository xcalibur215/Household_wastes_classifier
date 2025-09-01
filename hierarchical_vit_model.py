"""
Hierarchical Waste Classifier - Vision Transformer Model Architecture

This file contains the model architecture for the Hierarchical ViT used in waste classification.
Use this file to load the trained model weights (.pth file) in your web application.

Model Features:
- 30 fine-grained waste classes (specific waste types)
- 7 super categories (Metal, Paper, Glass, Plastic, Styrofoam, Organic, Textiles)
- Dual-head architecture with shared ViT backbone
- Progressive fine-tuning compatible

Author: Generated from waste_classifier_vit.ipynb
Date: August 31, 2025
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from timm import create_model
except ImportError as e:
    print(f"Required dependencies not found: {e}")
    print("Please install: pip install torch torchvision timm")
    raise


# Hierarchical Classification Mappings
SUPER_CLASSES = {
    0: [  # Metal_Aluminum
        'aerosol_cans',
        'aluminum_food_cans',
        'aluminum_soda_cans',
        'steel_food_cans'
    ],
    1: [  # Cardboard_Paper
        'cardboard_boxes',
        'cardboard_packaging',
        'magazines',
        'newspaper',
        'office_paper',
        'paper_cups'
    ],
    2: [  # Glass_Containers
        'glass_beverage_bottles',
        'glass_cosmetic_containers',
        'glass_food_jars'
    ],
    3: [  # Plastic_Items
        'disposable_plastic_cutlery',
        'plastic_cup_lids',
        'plastic_detergent_bottles',
        'plastic_food_containers',
        'plastic_shopping_bags',
        'plastic_soda_bottles',
        'plastic_straws',
        'plastic_trash_bags',
        'plastic_water_bottles'
    ],
    4: [  # Styrofoam_Products
        'styrofoam_cups',
        'styrofoam_food_containers'
    ],
    5: [  # Organic_Waste
        'coffee_grounds',
        'eggshells',
        'food_waste',
        'tea_bags'
    ],
    6: [  # Textiles_Clothing
        'clothing',
        'shoes'
    ]
}

SUPER_CLASS_NAMES = {
    0: 'Metal_Aluminum',
    1: 'Cardboard_Paper', 
    2: 'Glass_Containers',
    3: 'Plastic_Items',
    4: 'Styrofoam_Products',
    5: 'Organic_Waste',
    6: 'Textiles_Clothing'
}

# Build class name to index mapping (standard 30 classes)
CLASS_NAMES = [
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'clothing', 'coffee_grounds', 'disposable_plastic_cutlery',
    'eggshells', 'food_waste', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes', 'steel_food_cans',
    'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
]

# Create mappings
CLASS_TO_SUPER = {}
CLASS_IDX_TO_SUPER_IDX = {}

for super_id, class_list in SUPER_CLASSES.items():
    for class_name in class_list:
        CLASS_TO_SUPER[class_name] = super_id

for i, class_name in enumerate(CLASS_NAMES):
    if class_name in CLASS_TO_SUPER:
        CLASS_IDX_TO_SUPER_IDX[i] = CLASS_TO_SUPER[class_name]


class HierarchicalViT(nn.Module):
    """
    Hierarchical Vision Transformer for waste classification
    
    Architecture:
    - Shared ViT backbone (pretrained vit_small_patch16_224)
    - Dual prediction heads:
      • Fine-grained: 30 waste classes
      • Super-class: 7 categories (Metal, Paper, Glass, Plastic, Styrofoam, Organic, Textiles)
    
    Training Strategy:
    - Phase 1: Freeze backbone, train heads only
    - Phase 2: Fine-tune last transformer blocks + heads
    - Phase 3: Full model fine-tuning
    """
    
    def __init__(self, 
                 num_fine_classes=30, 
                 num_super_classes=7,
                 model_name='vit_small_patch16_224',
                 pretrained=True,
                 dropout=0.3):
        super().__init__()
        
        self.num_fine_classes = num_fine_classes
        self.num_super_classes = num_super_classes
        
        # Load pretrained ViT backbone
        self.backbone = create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0  # Remove classification head
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features  # 384 for vit_small
        
        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
        )
        
        # Fine-grained classification head (30 classes)
        self.fine_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_fine_classes)
        )
        
        # Super-class classification head (7 classes)
        self.super_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 4, num_super_classes)
        )
        
        # Initialize heads with Xavier initialization
        self._initialize_heads()
    
    def _initialize_heads(self):
        """Initialize classification heads"""
        for module in [self.fine_classifier, self.super_classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with dual outputs
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            fine_logits: Fine-grained predictions [batch_size, 30]
            super_logits: Super-class predictions [batch_size, 7]
        """
        # Extract features from ViT backbone
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # Process shared features
        processed_features = self.feature_processor(features)
        
        # Dual predictions
        fine_logits = self.fine_classifier(processed_features)     # [batch_size, 30]
        super_logits = self.super_classifier(processed_features)   # [batch_size, 7]
        
        return fine_logits, super_logits
    
    def freeze_backbone(self):
        """Freeze backbone parameters for Phase 1 training"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_last_blocks(self, num_blocks=2):
        """Unfreeze last transformer blocks for Phase 2 training"""
        # First freeze everything
        self.freeze_backbone()
        
        # Then unfreeze last few blocks
        if hasattr(self.backbone, 'blocks'):
            for block in self.backbone.blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Always unfreeze norm and head if they exist
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Get count of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params
    
    def predict_class_name(self, fine_idx):
        """Convert fine-grained class index to class name"""
        if 0 <= fine_idx < len(CLASS_NAMES):
            return CLASS_NAMES[fine_idx]
        return "Unknown"
    
    def predict_super_class_name(self, super_idx):
        """Convert super class index to super class name"""
        return SUPER_CLASS_NAMES.get(super_idx, "Unknown")
    
    def predict_from_fine_idx(self, fine_idx):
        """Get both fine and super class predictions from fine-grained index"""
        class_name = self.predict_class_name(fine_idx)
        super_idx = CLASS_IDX_TO_SUPER_IDX.get(fine_idx, 0)
        super_name = self.predict_super_class_name(super_idx)
        
        return {
            'fine_class': class_name,
            'fine_idx': fine_idx,
            'super_class': super_name,
            'super_idx': super_idx
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


class HierarchicalLoss(nn.Module):
    """
    Hierarchical loss combining fine-grained and super-class predictions
    """
    def __init__(self, 
                 fine_weight=0.7, 
                 super_weight=0.3,
                 num_super_classes=7,
                 focal_gamma=2.0):
        super().__init__()
        self.fine_weight = fine_weight
        self.super_weight = super_weight
        self.num_super_classes = num_super_classes
        
        # Use focal loss for both fine and super classification
        self.fine_loss = FocalLoss(gamma=focal_gamma, num_classes=30)
        self.super_loss = FocalLoss(gamma=focal_gamma, num_classes=num_super_classes)
        
    def forward(self, fine_logits, super_logits, fine_targets, super_targets):
        """
        Compute hierarchical loss
        
        Args:
            fine_logits: [batch_size, 30] fine-grained predictions
            super_logits: [batch_size, 7] super-class predictions
            fine_targets: [batch_size] fine-grained ground truth
            super_targets: [batch_size] super-class ground truth
        """
        # Ensure targets are long tensors
        fine_targets = fine_targets.long()
        super_targets = super_targets.long()
        
        # Compute individual losses
        loss_fine = self.fine_loss(fine_logits, fine_targets)
        loss_super = self.super_loss(super_logits, super_targets)
        
        # Weighted combination
        total_loss = self.fine_weight * loss_fine + self.super_weight * loss_super
        
        return total_loss, loss_fine, loss_super


def load_model(model_path, device='cpu', num_fine_classes=30, num_super_classes=7):
    """
    Load trained model from .pth file
    
    Args:
        model_path: Path to the .pth model file
        device: Device to load model on ('cpu', 'cuda', 'mps')
        num_fine_classes: Number of fine-grained classes (default: 30)
        num_super_classes: Number of super classes (default: 7)
    
    Returns:
        Loaded model ready for inference
    """
    # Create model instance
    model = HierarchicalViT(
        num_fine_classes=num_fine_classes,
        num_super_classes=num_super_classes,
        pretrained=False  # Don't load ImageNet weights when loading trained model
    )
    
    # Load trained weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    return model


def predict_single_image(model, image_tensor, device='cpu', return_probabilities=False):
    """
    Predict waste class for a single image
    
    Args:
        model: Loaded HierarchicalViT model
        image_tensor: Preprocessed image tensor [1, 3, 224, 224] or [3, 224, 224]
        device: Device to run inference on
        return_probabilities: Whether to return probability scores
    
    Returns:
        Dictionary with prediction results
    """
    model.eval()
    
    # Ensure correct tensor shape
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        fine_logits, super_logits = model(image_tensor)
        
        # Get predictions
        fine_probs = F.softmax(fine_logits, dim=1)
        super_probs = F.softmax(super_logits, dim=1)
        
        fine_pred = torch.argmax(fine_logits, dim=1).item()
        super_pred = torch.argmax(super_logits, dim=1).item()
        
        # Get class names
        fine_class_name = CLASS_NAMES[fine_pred]
        super_class_name = SUPER_CLASS_NAMES[super_pred]
        
        result = {
            'fine_class': fine_class_name,
            'fine_idx': fine_pred,
            'super_class': super_class_name,
            'super_idx': super_pred,
            'fine_confidence': fine_probs[0, fine_pred].item(),
            'super_confidence': super_probs[0, super_pred].item()
        }
        
        if return_probabilities:
            result['fine_probabilities'] = fine_probs[0].cpu().numpy()
            result['super_probabilities'] = super_probs[0].cpu().numpy()
        
        return result

try:
    from torchvision.transforms import v2
except ImportError:
    print("torchvision not found. Please install: pip install torchvision")
    v2 = None

def get_image_transform():
    """
    Get the image preprocessing transform used during training
    Apply this to input images before inference
    """
    if v2 is None:
        raise ImportError("torchvision not available. Please install: pip install torchvision")
        
    return v2.Compose([
        v2.Resize((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


if __name__ == "__main__":
    print("Hierarchical Waste Classifier - Model Architecture")
    print("=" * 50)
    
    model = HierarchicalViT()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randn(2, 3, 224, 224)
    fine_out, super_out = model(test_input)
    
    print(f"Fine-grained output shape: {fine_out.shape}")  # [2, 30]
    print(f"Super-class output shape: {super_out.shape}")   # [2, 7]
    
    # Display class information
    print(f"\nClass Information:")
    print(f"Fine-grained classes: {len(CLASS_NAMES)}")
    print(f"Super classes: {len(SUPER_CLASS_NAMES)}")
    print(f"Class mappings: {len(CLASS_IDX_TO_SUPER_IDX)} mapped")
    
    print("\nTo use this model in your web app:")
    print("1. Load model: model = load_model('best_hierarchical_model.pth', device)")
    print("2. Preprocess image: transform = get_image_transform()")
    print("3. Predict: result = predict_single_image(model, image_tensor)")
