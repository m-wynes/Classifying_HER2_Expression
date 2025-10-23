
# --- Import libraries ---
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split

# --- Define dataset class ---
class HER2_Expression_Data(Dataset):
    def __init__(self, img_dir, transform = None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = [f for f in os.listdir(img_dir) if f.endswith(".png")]
    
    def __len__ (self):
        return len(self.img_paths)
    
    #Extract HER2 label from the filename
    def _parse_label(self, filename):
        
        label_str = filename.split("_")[-1].replace(".png", "")
        
        if "0" in label_str:
            return 0
        elif "1+" in label_str:
            return 1
        elif "2+" in label_str:
            return 2
        elif "3+" in label_str:
            return 3
        else:
            raise ValueError(f"Unknown HER2 expression level: {filename}")
    
    # loading each image
    def __getitem__(self, idx):
        img_name = self.img_paths[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        label = self._parse_label(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --- Create DataLoaders ---
def get_dataloaders(train_directory, test_directory, batch_size = 32):
    # Transofrm images to 299 x 299
    transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])
    
    training_dataset = HER2_Expression_Data(img_dir = train_directory, transform = transform)
    testing_dataset = HER2_Expression_Data(img_dir = test_directory, transform = transform)
    
    # Train and validation split 
    val_size = int(0.2 * len(training_dataset)) # 20% of training data
    train_size = len(training_dataset) - val_size 
    
    train_data, val_data = random_split(training_dataset, [train_size, val_size])
    
    # DataLoader creation
    train_loader = DataLoader(
        train_data,
        batch_size = 32,
        shuffle = True,
        num_workers = 0,
        )
    
    val_loader = DataLoader(
        val_data,
        batch_size = 32,
        shuffle = False,
        num_workers = 0,
        )
    
    test_loader = DataLoader(
        testing_dataset,
        batch_size = 32,
        shuffle = False,
        num_workers = 0,
        )
    
    return train_loader, val_loader, test_loader