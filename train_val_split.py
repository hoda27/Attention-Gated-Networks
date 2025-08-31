import os
import shutil
from sklearn.model_selection import train_test_split

# Paths
images_dir = "C:\\Users\\hoda2\\Documents\\MastersGIUBerlin\\PanSegNet\\osfstorage-archive\\Pancreas_MRI_Dataset\\t1\\t1\\train\\imagesTr"
labels_dir = "C:\\Users\\hoda2\\Documents\\MastersGIUBerlin\\PanSegNet\\osfstorage-archive\\Pancreas_MRI_Dataset\\t1\\t1\\train\\labelsTr"
output_dir = "C:\\Users\\hoda2\\Documents\\MastersGIUBerlin\\code\\Attention-Gated-Networks\\dataio\\t1"

train_img_dir = os.path.join(output_dir, "train", "images")
train_lbl_dir = os.path.join(output_dir, "train", "masks")
val_img_dir   = os.path.join(output_dir, "val", "images")
val_lbl_dir   = os.path.join(output_dir, "val", "masks")
test_img_dir  = os.path.join(output_dir, "test", "images")
test_lbl_dir  = os.path.join(output_dir, "test", "masks")

# Create directories
for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
    os.makedirs(d, exist_ok=True)

# Collect image files
images = sorted(os.listdir(images_dir))

# Split
train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

# Copy files
for img in train_imgs:
    base = os.path.splitext(img)[0]
    parts = base.split("_")          
    base_name = "_".join(parts[:2]) 
    shutil.copy(os.path.join(images_dir, img), os.path.join(train_img_dir, img))
    shutil.copy(os.path.join(labels_dir, base_name + ".nii"+".gz"), os.path.join(train_lbl_dir, base_name + ".nii"+".gz"))

for img in val_imgs:
    base = os.path.splitext(img)[0]
    parts = base.split("_")        
    base_name = "_".join(parts[:2]) 
    shutil.copy(os.path.join(images_dir, img), os.path.join(val_img_dir, img))
    shutil.copy(os.path.join(labels_dir, base_name + ".nii"+".gz"), os.path.join(val_lbl_dir, base_name + ".nii"+".gz"))

for img in test_imgs:
    base = os.path.splitext(img)[0]
    parts = base.split("_")         
    base_name = "_".join(parts[:2]) 
    shutil.copy(os.path.join(images_dir, img), os.path.join(test_img_dir, img))
    shutil.copy(os.path.join(labels_dir, base_name + ".nii"+".gz"), os.path.join(test_lbl_dir, base_name + ".nii"+".gz"))

print("✅ Images and labels split into train/val successfully!")