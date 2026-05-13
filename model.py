import torch
import torch.nn as nn
from ultralytics import YOLO

class MultiTaskYOLOv8(nn.Module):
    def __init__(self, num_detection_classes=2):
        super(MultiTaskYOLOv8, self).__init__()
        
        # Load the official pretrained YOLOv8 Nano model
        self.yolo_model = YOLO("yolov8n.pt").model  
        
        # Design our Custom Environment Classification Head
        # Stage 9 of the backbone has 256 channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),                 
            nn.Linear(128, 1)                
        )

        self.backbone_features = None
        self.trainer = None  # Will be set dynamically by the trainer
        self.criterion = None  # Will be set dynamically by the trainer

        # Register hook to capture Stage 9 features
        self.yolo_model.model[9].register_forward_hook(self._hook_fn)
        self.end2end = False

    def _hook_fn(self, module, input, output):
        self.backbone_features = output

    def forward(self, x, *args, **kwargs):
        """
        Handles training batches from the trainer OR raw image tensors during inference.
        """
        is_training_batch = isinstance(x, dict)
        images = x["img"] if is_training_batch else x

        # Ensure the inner YOLO model is in inference mode to get the correct output format
        # for validation/NMS, but keep gradients enabled if we are training.
        was_training = self.yolo_model.training
        self.yolo_model.training = False
        
        try:
            with torch.set_grad_enabled(was_training or self.training):
                det_output = self.yolo_model(images, *args, **kwargs)
        finally:
            self.yolo_model.training = was_training

        # Run environment classification head
        features = self.backbone_features 
        pooled = self.avgpool(features)                 
        pooled = torch.flatten(pooled, start_dim=1)     
        env_logits = self.classification_head(pooled)   
        env_logits = env_logits.squeeze(1)              

        predictions = (det_output, env_logits)

        # 1. TRAINING PATH: Return loss directly to the trainer
        if is_training_batch and self.training and self.trainer is not None:
            return self.trainer.compute_loss(x, predictions)

        # 2. VALIDATION/INFERENCE PATH: Return ONLY the detection tensor
        # Standard Ultralytics post-processing (NMS) cannot handle the multi-task tuple.
        # We return det_output alone to satisfy the validator's expectations.
        return det_output

    def loss(self, batch, preds):
        # preds is now just det_output because of the forward() change above
        det_preds = preds 

        criterion = getattr(self, "criterion", None)
        if criterion is None and self.trainer is not None:
            criterion = getattr(self.trainer, "criterion", None)

        if criterion is not None:
            loss_sum, loss_items = criterion(det_preds, batch)
            # Return 4 elements to match the trainer's internal tracking
            pad = torch.zeros(1, device=loss_items.device)
            return loss_sum, torch.cat([loss_items.detach(), pad])
            
        return torch.zeros(1, device=batch["img"].device), torch.zeros(4, device=batch["img"].device)
    
    def state_dict(self, *args, **kwargs):
        """
        Intercepts state_dict calls during checkpoint saving to temporarily
        remove unpicklable trainer/validator references, preventing serialization crashes.
        """
        # 1. Back up the live references
        backup_trainer = getattr(self, "trainer", None)
        backup_validator = None
        
        # Also clean up the validator reference if it managed to attach here
        if backup_trainer and hasattr(backup_trainer, "validator"):
            backup_validator = backup_trainer.validator
            backup_trainer.validator = None

        self.trainer = None

        try:
            # 2. Call the real PyTorch state_dict generation
            return super().state_dict(*args, **kwargs)
        finally:
            # 3. Restore everything seamlessly for the next training epoch
            if backup_trainer is not None:
                self.trainer = backup_trainer
                if backup_validator is not None:
                    self.trainer.validator = backup_validator