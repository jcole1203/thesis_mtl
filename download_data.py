import os
import shutil
from roboflow import Roboflow

# Initialize Roboflow (Keep your actual API key here)
rf = Roboflow(api_key="NPPY8jM3eUZtIcvJJeY9")

def move_to_multitask(src_dir, dest_dir):
    """Utility to move downloaded images and labels into our structured folders."""
    src_images = os.path.join(src_dir, "images")
    src_labels = os.path.join(src_dir, "labels")
    
    dest_images = os.path.join(dest_dir, "images")
    dest_labels = os.path.join(dest_dir, "labels")

    if os.path.exists(src_images):
        os.makedirs(dest_images, exist_ok=True)
        for f in os.listdir(src_images):
            shutil.move(os.path.join(src_images, f), os.path.join(dest_images, f))
            
    if os.path.exists(src_labels):
        os.makedirs(dest_labels, exist_ok=True)
        for f in os.listdir(src_labels):
            shutil.move(os.path.join(src_labels, f), os.path.join(dest_labels, f))

def download_and_route_data():
    dest_indoor_train = "./fire_multitask_data/train/indoor"
    dest_indoor_val = "./fire_multitask_data/val/indoor"
    dest_outdoor_train = "./fire_multitask_data/train/outdoor"
    dest_outdoor_val = "./fire_multitask_data/val/outdoor"

    # ==========================================
    # TASK A: INDOOR (Skip if already populated)
    # ==========================================
    # We check if images already exist so we don't waste bandwidth re-downloading
    if os.path.exists(dest_indoor_train + "/images") and os.listdir(dest_indoor_train + "/images"):
        print("✅ Indoor Fire Dataset already downloaded and organized. Skipping...")
    else:
        print("Downloading Indoor Fire Dataset...")
        try:
            project_indoor = rf.workspace("aj-garcia-736tc").project("fire-dataset-for-yolov8")
            dataset_indoor = project_indoor.version(10).download("yolov8", location="./temp_indoor")
            move_to_multitask("./temp_indoor/train", dest_indoor_train)
            move_to_multitask("./temp_indoor/valid", dest_indoor_val)
            shutil.rmtree("./temp_indoor", ignore_errors=True)
            print("🎉 Indoor dataset organized!")
        except Exception as e:
            print(f"Indoor Download Failed: {e}")

    # ==========================================
    # TASK B: OUTDOOR (METU Public Dataset)
    # ==========================================
    print("Downloading Outdoor Fire & Smoke Dataset...")
    try:
        # Middle East Tech University Public Dataset (Highly stable and open API access)
        project_outdoor = rf.workspace("middle-east-tech-university").project("fire-and-smoke-detection-hiwia")
        dataset_outdoor = project_outdoor.version(2).download("yolov8", location="./temp_outdoor")

        move_to_multitask("./temp_outdoor/train", dest_outdoor_train)
        
        # Check for 'valid', 'test', or 'test' folders to route validation set
        src_val = "./temp_outdoor/valid" if os.path.exists("./temp_outdoor/valid") else "./temp_outdoor/test"
        move_to_multitask(src_val, dest_outdoor_val)
        
        shutil.rmtree("./temp_outdoor", ignore_errors=True)
        print("🎉 Outdoor Dataset successfully built and routed!")
    except Exception as e:
        print(f"Outdoor Download Failed: {e}")

if __name__ == "__main__":
    download_and_route_data()