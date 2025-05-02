import argparse
import os
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Replace with actual generator and discriminator class imports
from models import CEGenerator, CEDiscriminator


def create_single_image_dataloader(image_path, transform, input_size, output_size, batch_size=1):
    class SingleImageDataset(Dataset):
        def __init__(self, image_path, transform):
            self.image_path = image_path
            self.transform = transform

        def __getitem__(self, index):
            img = Image.open(self.image_path).convert('RGB')
            img = self.transform(img)
            masked_img, _ = self.apply_center_mask(img)
            return img, masked_img, torch.zeros_like(img)

        def apply_center_mask(self, img):
            i = (output_size - input_size) // 2
            masked_img = img.clone()
            masked_img[:, :i, :] = 1
            masked_img[:, -i:, :] = 1
            masked_img[:, :, :i] = 1
            masked_img[:, :, -i:] = 1
            return masked_img, None

        def __len__(self):
            return 1

    dataset = SingleImageDataset(image_path, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def generate_html(output_dir):
    html_path = os.path.join(output_dir, "index.html")
    html_content = f"""
    <html>
    <head>
        <title>Inpainting Result</title>
        <style>
            body {{ font-family: sans-serif; text-align: center; padding: 20px; }}
            img {{ width: 256px; margin: 10px; border: 2px solid #ccc; }}
            .row {{ display: flex; justify-content: center; flex-wrap: wrap; }}
            .caption {{ margin-top: 5px; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h2>Image Inpainting Visualization</h2>
        <div class="row">
            <div>
                <img src="images/val_0_masked.jpg" alt="Masked">
                <div class="caption">Masked Input</div>
            </div>
            <div>
                <img src="images/val_0_result.jpg" alt="Output">
                <div class="caption">Generated Output</div>
            </div>
            <div>
                <img src="images/val_0_truth.jpg" alt="Original">
                <div class="caption">Original Image</div>
            </div>
        </div>
    </body>
    </html>
    """
    with open(html_path, 'w') as f:
        f.write(html_content)
    print(f'Generated visualization at: {html_path}')


def generate_html_for_single_image(G_net, D_net, device, image_path, output_dir, transform, input_size, output_size):
    G_net.eval()
    D_net.eval()
    torch.set_grad_enabled(False)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(os.path.join(output_dir, 'images'))

    data_loader = create_single_image_dataloader(image_path, transform, input_size, output_size)
    imgs, masked_imgs, _ = next(iter(data_loader))
    masked_imgs = masked_imgs.to(device)
    outputs = G_net(masked_imgs)
    masked_imgs = masked_imgs.cpu()
    results = outputs.cpu()

    save_image(masked_imgs[0], os.path.join(output_dir, 'images/val_0_masked.jpg'))
    save_image(results[0], os.path.join(output_dir, 'images/val_0_result.jpg'))
    save_image(imgs[0], os.path.join(output_dir, 'images/val_0_truth.jpg'))

    generate_html(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run image inpainting on a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, default='./generator_final.pt', help='Path to the trained generator .pt file')
    parser.add_argument('--output_dir', type=str, default='./test', help='Directory to save HTML output')
    parser.add_argument('--input_size', type=int, default=128, help='Input mask size')
    parser.add_argument('--output_size', type=int, default=192, help='Output (image) size')

    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((args.output_size, args.output_size)),
        transforms.CenterCrop(args.output_size),
        transforms.ToTensor()
    ])

    G_net = CEGenerator(extra_upsample=True).to(device)
    G_net.load_state_dict(torch.load(args.model_path, map_location=device))

    D_net = CEDiscriminator().to(device)

    generate_html_for_single_image(
        G_net, D_net, device, args.image_path,
        args.output_dir, transform,
        args.input_size, args.output_size
    )
