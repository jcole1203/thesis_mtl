import cv2
import torch
from ultralytics import YOLO

torch.serialization.add_safe_globals(["MultiTaskYOLOv8"])

def test_fire_model(source_path, weights_path="runs/detect/train-69/weights/best.pt"):
    print("\n🔨 FORCING MANUAL PYTHON FILTER HEAD...")
    model = YOLO(weights_path)
    
    # Let the model output everything natively without arguments restrictions
    results = model.predict(source=source_path, save=False, verbose=False)
    
    for result in results:
        img = result.orig_img.copy()
        
        detected_context = "Context: Undetermined"
        highest_env_conf = 0.0
        fire_boxes_count = 0
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            
            # 1️⃣ STEP 1: Process Background Context Text (Classes 1 and 2)
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id in [1, 2]:
                    if conf > highest_env_conf:
                        highest_env_conf = conf
                        detected_context = "Context: Indoor" if cls_id == 1 else "Context: Outdoor"
                        detected_context += f" ({conf:.2f})"
            
            # 2️⃣ STEP 2: MANUALLY FILTER AND DRAW FIRE BOXES (Class 0)
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == 0:
                    # 🎯 CRITICAL SWEET SPOT: Only allow boxes with real, strong confidence
                    if conf < 0.25:
                        continue # Drop the weak structural background noise!
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Eliminate any absurd full-image bounding boxes manually
                    box_width = x2 - x1
                    box_height = y2 - y1
                    if box_height > (img.shape[0] * 0.85): 
                        continue # Blocks the giant vertical pillars cutting the sky

                    fire_boxes_count += 1
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, f"Fire: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Force render the highest recorded context regardless of box count
        cv2.putText(img, detected_context, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        print(f"📊 Manual filter complete. Placed {fire_boxes_count} pristine fire targets.")
        cv2.imshow("Thesis Multi-Task Fire Detection Test", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_fire_model("outdoor.jpg", weights_path="runs/detect/train-45/weights/best.pt")