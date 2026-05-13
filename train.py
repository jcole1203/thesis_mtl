import os
import torch
import torch.nn as nn
from copy import copy
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

# Import your custom modules
from dataset import MultiTaskFireDataset, collate_fn
from model import MultiTaskYOLOv8
from ultralytics.models.yolo.detect import DetectionValidator

class MultiTaskValidator(DetectionValidator):
    """
    Custom validator that handles the specific metadata mapping 
    and 4-element loss requirements of the MultiTask model.
    """
    def preprocess(self, batch):
        # 1. Map the keys for the validator
        batch["img"] = batch["images"]
        batch["bboxes"] = batch["bbox_targets"][:, 2:]    
        batch["cls"] = batch["bbox_targets"][:, 1:2]      
        batch["batch_idx"] = batch["bbox_targets"][:, 0].view(-1) 
        
        # 2. Add metadata for coordinate scaling
        h, w = batch["img"].shape[2:]
        batch["ori_shape"] = [(h, w)] * batch["img"].shape[0]
        batch["imgsz"] = (h, w)
        batch["ratio_pad"] = [(1.0, 1.0, 0.0, 0.0)] * batch["img"].shape[0]
        batch["im_file"] = [""] * batch["img"].shape[0]
        
        return super().preprocess(batch)

    def __call__(self, trainer=None, model=None):
        """
        Custom call to ensure the model being validated always 
        returns 4 loss items, safely ignoring string paths during final eval.
        """
        val_model = model or getattr(self, "model", None)
        
        # FIX: Ensure val_model is a real PyTorch module, not a path string
        if isinstance(val_model, nn.Module) and not getattr(val_model, "_loss_is_patched", False):
            raw_model = val_model.module if hasattr(val_model, "module") else val_model
            
            # Define the picklable patch
            def patched_loss(v_batch, v_preds):
                det_preds = v_preds[0] if isinstance(v_preds, tuple) else v_preds
                criterion = getattr(raw_model, "criterion", None)
                if criterion is None and trainer is not None:
                    criterion = getattr(trainer, "criterion", None)
                
                if criterion is not None:
                    loss_sum, loss_items = criterion(det_preds, v_batch)
                    pad = torch.zeros(1, device=loss_items.device)
                    return loss_sum, torch.cat([loss_items.detach(), pad])
                return torch.zeros(1, device=v_batch["img"].device), torch.zeros(4, device=v_batch["img"].device)

            raw_model.loss = patched_loss
            val_model._loss_is_patched = True

        return super().__call__(trainer, model)

class MultiTaskTrainer(DetectionTrainer):
    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        
        # Define your thesis detection classes directly here
        self.custom_class_names = {
            0: "fire"
        }
        
        

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Initializes our custom MultiTaskYOLOv8 architecture."""
        model = MultiTaskYOLOv8(weights or self.args.model)
        model.trainer = self
        
        # 1. Force the model wrapper to tell Ultralytics it's NOT an end-to-end model
        model.end2end = False
        
        # 2. 🔥 REWRITE DETECTOR NAMES DIRECTLY HERE 🔥
        # This replaces the default 'person', 'bicycle', etc., without crashing dataset configurations
        model.names = {0: "fire"}
        if hasattr(model, 'yolo_model'):
            model.yolo_model.names = {0: "fire"}
        
        # 3. Safe assignment for loss function
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        
        if hasattr(self, 'pos_weight') and self.pos_weight is not None:
            self.classification_loss_fn.pos_weight = self.pos_weight.to(self.device)
            
        return model

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Constructs the custom PyTorch DataLoader required by MultiTaskFireDataset."""
        import os
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.join(script_dir, "fire_multitask_data")
        
        dataset = MultiTaskFireDataset(
            root_dir=base_path, 
            split=mode
        )
        
        # 🟢 DIRECT EXPLICIT PATH TARGETING
        if len(dataset) == 0:
            target_split = "val" if mode in ["val", "valid"] else "train"
            split_folder = os.path.join(base_path, target_split)
            
            found_images = []
            found_labels = []
            found_annotations = []
            
            # Define explicit targets matching your exact tree layout
            environments = {
                "indoor": 0,
                "outdoor": 1
            }
            
            for env_name, env_cls in environments.items():
                # Force Python directly into the 'images' subdirectory where your files live
                images_target_dir = os.path.join(split_folder, env_name, "images")
                
                if os.path.exists(images_target_dir):
                    for file in os.listdir(images_target_dir):
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                            full_path = os.path.join(images_target_dir, file)
                            found_images.append(full_path)
                            found_labels.append(env_cls)
                            found_annotations.append({"cls_target": env_cls, "img_path": full_path})
            
            if found_images:
                print(f"📁 Explicit Mapping Success! Loaded {len(found_images)} images for split context ({mode}).")
                
                # Bind arrays directly to the dataset instance attributes
                for attr in ['img_paths', 'images', 'imgs', 'file_list']:
                    if hasattr(dataset, attr):
                        setattr(dataset, attr, found_images)
                
                if not hasattr(dataset, 'img_paths') or not getattr(dataset, 'img_paths'):
                    dataset.img_paths = found_images
                    
                dataset.labels = found_labels
                dataset.annotations = found_annotations
                
                # Update Python wrapper length calculation
                dataset.__len__ = lambda: len(found_images)
            else:
                raise RuntimeError(
                    f"❌ Dataset Critical Failure: Targeted image folders inside '{split_folder}' are empty."
                )
        
        if mode == "train":
            self.pos_weight = torch.tensor([4.0], dtype=torch.float32)
            if hasattr(self, 'classification_loss_fn') and self.classification_loss_fn is not None:
                self.classification_loss_fn.pos_weight = self.pos_weight.to(self.device)
                print(f"⚖️ Applied custom environment class balance weights: {self.pos_weight.item():.4f}")

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True) if rank != -1 else None
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(sampler is None and mode == "train"),
            num_workers=0,
            collate_fn=collate_fn,
            sampler=sampler
        )
        
        return loader

    def preprocess_batch(self, batch):
        """Map training keys to what Ultralytics expect"""
        batch["img"] = batch["images"]
        batch["bboxes"] = batch["bbox_targets"][:, 2:]    
        batch["cls"] = batch["bbox_targets"][:, 1:2]      
        batch["batch_idx"] = batch["bbox_targets"][:, 0].view(-1) 
        
        # Add metadata for the validator/logger
        if "ori_shape" not in batch:
            h, w = batch["img"].shape[2:]
            batch["ori_shape"] = [(h, w)] * batch["img"].shape[0]
            # NEW: ratio_pad = (h_ratio, w_ratio, pad_h, pad_w)
            batch["ratio_pad"] = [(1.0, 1.0, 0.0, 0.0)] * batch["img"].shape[0]
            
        return super().preprocess_batch(batch)

    def get_validator(self):
        """Returns the custom picklable MultiTaskValidator."""
        self.loss_names = 'Box', 'Class', 'DFL', 'Env'
        
        
        return MultiTaskValidator(
            dataloader=self.test_loader, 
            save_dir=self.save_dir, 
            args=copy(self.args)
        )
        
        def patched_preprocess(batch):
            batch["img"] = batch["images"]
            batch["bboxes"] = batch["bbox_targets"][:, 2:]    
            batch["cls"] = batch["bbox_targets"][:, 1:2]      
            batch["batch_idx"] = batch["bbox_targets"][:, 0].view(-1) 
            
            # Metadata keys to prevent KeyError crashes
            h, w = batch["img"].shape[2:]
            batch["ori_shape"] = [(h, w)] * batch["img"].shape[0]
            batch["imgsz"] = (h, w)
            
            # NEW: ratio_pad handles the coordinate scaling for metrics
            # Format: (width_ratio, height_ratio, width_pad, height_pad)
            batch["ratio_pad"] = [(1.0, 1.0, 0.0, 0.0)] * batch["img"].shape[0]
            
            # Optional: Some versions check for 'im_file' for plotting
            batch["im_file"] = [""] * batch["img"].shape[0]
            
            # ... (Keep your existing model.loss patch below) ...
            if hasattr(validator, 'model') and validator.model is not None:
                # [Previous safe_validator_loss logic here]
                pass

            return original_preprocess(batch)
        
        validator.preprocess = patched_preprocess
        return validator

    def plot_training_labels(self):
        pass

    def plot_training_samples(self, batch, ni):
        pass

    def compute_loss(self, batch, predictions):
        """Override compute_loss to combine YOLO detection loss & environment classification loss"""
        det_preds, env_logits = predictions
        
        # 1. Pure PyTorch unwrapping: extract raw model if wrapped in DDP or DataParallel
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # 2. Initialize YOLO criterion dynamically using the unwrapped inner model
        if not hasattr(self, "criterion") or self.criterion is None:
            raw_model.yolo_model.hyp = self.args
            self.criterion = v8DetectionLoss(raw_model.yolo_model)
            self.criterion.hyp = self.args
            raw_model.criterion = self.criterion

        # 3. Double check and explicitly overwrite internal loss config parameters 
        self.criterion.hyp = self.args
        raw_model.yolo_model.hyp = self.args
        
        # 4. Compute native YOLOv8 loss (L_det)
        yolo_loss_sum, yolo_loss_components = self.criterion(det_preds, batch)

        # 5. Extract environment labels from our custom batch mapping
        env_targets = batch["env_labels"].to(self.device).float()

        # 6. Compute Environment Binary Cross BCE Loss (L_env)
        loss_env = self.classification_loss_fn(env_logits, env_targets)

        # 7. ⚖️ METHODOLOGY ALIGNMENT: Explicit Lambda task scaling weight (Section 3.7) ⚖️
        self.loss_lambda = 1.0  
        weighted_loss_env = self.loss_lambda * loss_env

        # 8. Combine losses according to: L_total = L_det + lambda * L_env
        total_loss = yolo_loss_sum + weighted_loss_env

        # 9. Detach loss components for training tracking outputs (Box, Class, DFL, Env)
        loss_items = torch.cat((yolo_loss_components, loss_env.unsqueeze(0))).detach()

        return total_loss, loss_items
    
    def save_model(self):
        """
        Intercepts checkpoint saving to strip unpicklable references 
        from both the main model and EMA, ensuring safe serialization.
        """
        # 1. Gather all potential model instances where the trainer might be attached
        models_to_strip = []
        if hasattr(self, 'model') and self.model is not None:
            models_to_strip.append(self.model)
        if hasattr(self, 'ema') and self.ema is not None and hasattr(self.ema, 'ema'):
            models_to_strip.append(self.ema.ema)

        # 2. Sever the trainer links and back them up
        for m in models_to_strip:
            # Unravel DistributedDataParallel wrappers if any exist
            raw_model = m.module if hasattr(m, 'module') else m
            if hasattr(raw_model, 'trainer'):
                raw_model.trainer = None

        try:
            # 3. Call Ultralytics' native saving mechanism (now safely decoupled!)
            return super().save_model()
        finally:
            # 4. Reattach the trainer reference so training can proceed uninterrupted
            for m in models_to_strip:
                raw_model = m.module if hasattr(m, 'module') else m
                raw_model.trainer = self

# ==========================================================
# 🏃 TRAINING RUNNER
# ==========================================================
import torch
from ultralytics import YOLO
# Import your custom class so the script knows what it is
# (Assuming MultiTaskYOLOv8 is defined in this file or imported)

if __name__ == "__main__":
    print("🛡️ Patching PyTorch Security & Metadata...")

    # 1. TELL PYTORCH TO TRUST YOUR CUSTOM CLASS
    # This fixes the 'Weights only load failed' error
    torch.serialization.add_safe_globals(["MultiTaskYOLOv8"])
    
    ckpt_path = "runs/detect/train-69/weights/last.pt"

    # 2. LOAD THE CHECKPOINT
    # We use weights_only=False because it's your own trusted file
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_from_ckpt = checkpoint['model']

    # 3. ATTACH THE MISSING BLUEPRINT
    # This fixes the 'AttributeError: yaml'
    if not hasattr(model_from_ckpt, 'yaml'):
        # We give it a dummy config that tells YOLO there is 1 class (Fire)
        model_from_ckpt.yaml = {'nc': 1, 'names': {0: 'fire'}, 'yaml': 'fire_data.yaml'}
    
    overrides = {
        "model": ckpt_path,
        "data": "fire_data.yaml",
        "epochs": 50,
        "imgsz": 640,
        "resume": False, 
    }

    # 4. INITIALIZE AND INJECT
    trainer = MultiTaskTrainer(overrides=overrides)
    trainer.model = model_from_ckpt.to("cuda")
    
    print("✅ System Ready. Resuming from Epoch 40...")
    trainer.train()