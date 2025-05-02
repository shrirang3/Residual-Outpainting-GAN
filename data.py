import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, root_dir, input_size=128, output_size=192, outpaint=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            input_size (int): Size of the input image (center crop).
            output_size (int): Size of the output image.
            outpaint (bool): If True, perform outpainting. If False, perform inpainting.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.input_size = input_size
        self.output_size = output_size
        self.outpaint = outpaint
        
        # Store transform as a composition of transforms rather than a module
        # This helps avoid pickling issues
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((output_size, output_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transform
            
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and 
                           (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Resize image to output_size
        image = transforms.Resize((self.output_size, self.output_size))(image)
        
        # Apply transformations
        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)
        
        # Create masked image (center portion for inpainting, outer portion for outpainting)
        masked_img = img_tensor.clone()
        
        if self.outpaint:
            # For outpainting: mask (set to 0) the outer region, keep only the center
            # Create a mask that's 1 in the center and 0 elsewhere
            mask = torch.zeros_like(img_tensor)
            start = (self.output_size - self.input_size) // 2
            end = start + self.input_size
            mask[:, start:end, start:end] = 1.0
            
            # Apply mask to create input image (center only)
            masked_img = img_tensor * mask
            
            # The masked part is the entire image (for comparison in loss)
            masked_part = img_tensor
        else:
            # For inpainting: mask (set to 0) the center region, keep only the outer
            # Create a mask that's 0 in the center and 1 elsewhere
            mask = torch.ones_like(img_tensor)
            start = (self.output_size - self.input_size) // 2
            end = start + self.input_size
            mask[:, start:end, start:end] = 0.0
            
            # Apply mask to create input image (outer only)
            masked_img = img_tensor * mask
            
            # The masked part is just the center (for comparison in loss)
            masked_part = img_tensor[:, start:end, start:end]
        
        return img_tensor, masked_img, masked_part


def get_data_loaders(train_dir, val_dir, test_dir, batch_size=4, input_size=128, output_size=192, 
                    outpaint=True, transform=None, num_workers=0):
    """
    Create and return data loaders for train, validation, and test sets.
    
    Args:
        train_dir (str): Directory with training images.
        val_dir (str): Directory with validation images.
        test_dir (str): Directory with test images.
        batch_size (int): Batch size for DataLoader.
        input_size (int): Size of the input image (center crop).
        output_size (int): Size of the output image.
        outpaint (bool): If True, perform outpainting. If False, perform inpainting.
        transform (callable, optional): Optional transform to be applied on samples.
        num_workers (int): Number of workers for DataLoader. Set to 0 to avoid multiprocessing issues.
    
    Returns:
        dict: Dictionary containing 'train', 'val', and 'test' DataLoaders.
    """
    data_loaders = {}
    
    # Create datasets
    if os.path.exists(train_dir) and len(os.listdir(train_dir)) > 0:
        train_data = ImageDataset(train_dir, input_size, output_size, outpaint, transform)
        data_loaders['train'] = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        print(f"Warning: Training directory '{train_dir}' is empty or does not exist.")
        data_loaders['train'] = None
    
    if os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0:
        val_data = ImageDataset(val_dir, input_size, output_size, outpaint, transform)
        data_loaders['val'] = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        print(f"Warning: Validation directory '{val_dir}' is empty or does not exist.")
        data_loaders['val'] = None
    
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_data = ImageDataset(test_dir, input_size, output_size, outpaint, transform)
        data_loaders['test'] = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        print(f"Warning: Test directory '{test_dir}' is empty or does not exist.")
        data_loaders['test'] = None
    
    return data_loaders