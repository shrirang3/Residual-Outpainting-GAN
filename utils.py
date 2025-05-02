import os
import torch
import numpy as np
import torchvision.utils as vutils
from torch.autograd import Variable

# Patch dimensions for the discriminator
patch_h, patch_w = 8, 8
patch = (1, patch_h, patch_w)

def is_power_two(n):
    """Check if n is a power of 2"""
    return (n & (n-1) == 0) and n != 0

def get_adv_weight(adv_weight, epoch, decay_rate=0.1, start_decay=50, end_decay=150):
    """Calculate the adversarial weight using a decay schedule."""
    if epoch < start_decay:
        return 0
    elif epoch > end_decay:
        return adv_weight
    else:
        return adv_weight * ((epoch - start_decay) / (end_decay - start_decay))

def generate_html(G_net, D_net, device, data_loaders, save_path, outpaint=True, num_images=8):
    """Generate HTML visualization of model output."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Create HTML header
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Outpainting Results</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 20px;}
            .example {margin-bottom: 40px; display: flex; align-items: center;}
            .image-container {margin-right: 20px; text-align: center;}
            .caption {margin-top: 5px; font-size: 14px;}
        </style>
    </head>
    <body>
        <h1>Image Outpainting Results</h1>
    """
    
    # Set models to eval mode
    G_net.eval()
    D_net.eval()
    
    # Generate examples from validation set
    with torch.no_grad():
        for batch_idx, (imgs, masked_imgs, masked_parts) in enumerate(data_loaders['val']):
            if batch_idx >= num_images // data_loaders['val'].batch_size + 1:
                break
            
            # Move data to device
            real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)
            masked_imgs = Variable(masked_imgs.type(torch.FloatTensor)).to(device)
            
            # Generate outputs
            gen_imgs = G_net(masked_imgs)
            
            # For each image in batch
            for i in range(min(real_imgs.size(0), num_images - batch_idx * data_loaders['val'].batch_size)):
                # Save images
                real_img = real_imgs[i].unsqueeze(0)
                masked_img = masked_imgs[i].unsqueeze(0)
                gen_img = gen_imgs[i].unsqueeze(0)
                
                # Create comparison images
                comparison = torch.cat([masked_img, gen_img, real_img], dim=0)
                comparison_path = os.path.join(save_path, f"comparison_{batch_idx}_{i}.png")
                vutils.save_image(comparison, comparison_path, normalize=True, nrow=3)
                
                # Add to HTML
                html_content += f"""
                <div class="example">
                    <div class="image-container">
                        <img src="{os.path.basename(comparison_path)}" width="800">
                        <div class="caption">Left: Input | Middle: Generated | Right: Ground Truth</div>
                    </div>
                </div>
                """
    
    # Close HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write HTML file
    with open(os.path.join(save_path, "results.html"), "w") as f:
        f.write(html_content)
    
    # Set models back to their previous mode
    G_net.train()
    D_net.train()