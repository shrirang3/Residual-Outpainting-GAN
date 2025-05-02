import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
import pickle
from collections import defaultdict

from models import CEGenerator, CEDiscriminator, weights_init_normal, VGGLoss
from data import get_data_loaders
from utils import is_power_two, generate_html, get_adv_weight, patch_h, patch_w, patch


def train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
             data_loaders, model_save_path, html_save_path, n_epochs=251, start_epoch=0, 
             outpaint=True, adv_weight=0.001, save_interval=10):
    '''
    Based on Context Encoder implementation in PyTorch.
    '''
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor
    hist_loss = defaultdict(list)

    # Add perceptual loss
    criterion_perceptual = VGGLoss(device)
    perceptual_weight = 0.1  # Weight for perceptual loss

    for epoch in range(start_epoch, n_epochs):

        for phase in ['train', 'val']:
            # Skip this phase if no data loader is available or it's empty
            if not data_loaders.get(phase) or isinstance(data_loaders[phase], list):
                print(f"Skipping {phase} phase - no data available")
                continue
                
            batches_done = 0

            running_loss_pxl = 0.0
            running_loss_adv = 0.0
            running_loss_D = 0.0
            running_loss_perceptual = 0.0

            for idx, (imgs, masked_imgs, masked_parts) in enumerate(data_loaders[phase]):
                if phase == 'train':
                    G_net.train()
                    D_net.train()
                else:
                    G_net.eval()
                    D_net.eval()
                torch.set_grad_enabled(phase == 'train')

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False).to(device)
                fake = Variable(Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False).to(device)
                # Configure input
                imgs = Variable(imgs.type(Tensor)).to(device)
                masked_imgs = Variable(masked_imgs.type(Tensor)).to(device)
                if not(outpaint):
                    masked_parts = Variable(masked_parts.type(Tensor)).to(device)

                # -----------
                #  Generator
                # -----------
                if phase == 'train':
                    optimizer_G.zero_grad()
                # Generate a batch of images
                outputs = G_net(masked_imgs)
                # Adversarial and pixelwise loss
                if not(outpaint):
                    loss_pxl = criterion_pxl(outputs, masked_parts) # inpaint: compare center part only
                    loss_perceptual = criterion_perceptual(outputs, masked_parts)
                else:
                    loss_pxl = criterion_pxl(outputs, imgs) # outpaint: compare to full ground truth
                    loss_perceptual = criterion_perceptual(outputs, imgs)
                loss_adv = criterion_D(D_net(outputs), valid)
                # Total loss
                cur_adv_weight = get_adv_weight(adv_weight, epoch)
                loss_G = (1 - cur_adv_weight) * loss_pxl + cur_adv_weight * loss_adv + perceptual_weight * loss_perceptual
                if phase == 'train':
                    loss_G.backward()
                    optimizer_G.step()

                # ---------------
                #  Discriminator
                # ---------------
                if phase == 'train':
                    optimizer_D.zero_grad()
                # Measure discriminator's ability to classify real from generated samples
                if not(outpaint):
                    real_loss = criterion_D(D_net(masked_parts), valid) # inpaint: check center part only
                else:
                    real_loss = criterion_D(D_net(imgs), valid) # outpaint: check full ground truth
                fake_loss = criterion_D(D_net(outputs.detach()), fake)
                loss_D = 0.5 * (real_loss + fake_loss)
                if phase == 'train':
                    loss_D.backward()
                    optimizer_D.step()

                # Update & print statistics
                batches_done += 1
                running_loss_pxl += loss_pxl.item()
                running_loss_adv += loss_adv.item()
                running_loss_D += loss_D.item()
                running_loss_perceptual += loss_perceptual.item()
                if phase == 'train' and is_power_two(batches_done):
                    print('Batch {:d}/{:d}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}  loss_perceptual {:.4f}'.format(
                          batches_done, len(data_loaders[phase]), loss_pxl.item(), loss_adv.item(), loss_D.item(), loss_perceptual.item()))

            # Store model & visualize examples
            if phase == 'train' and epoch % save_interval == 0:
                if not os.path.exists(model_save_path):
                    os.makedirs(model_save_path)
                torch.save(G_net.state_dict(), os.path.join(model_save_path, f'G_{epoch}.pt'))
                torch.save(D_net.state_dict(), os.path.join(model_save_path, f'D_{epoch}.pt'))
                generate_html(G_net, D_net, device, data_loaders, os.path.join(html_save_path, str(epoch)), outpaint=outpaint)

            # Store & print statistics
            cur_loss_pxl = running_loss_pxl / batches_done
            cur_loss_adv = running_loss_adv / batches_done
            cur_loss_D = running_loss_D / batches_done
            cur_loss_perceptual = running_loss_perceptual / batches_done
            hist_loss[phase + '_pxl'].append(cur_loss_pxl)
            hist_loss[phase + '_adv'].append(cur_loss_adv)
            hist_loss[phase + '_D'].append(cur_loss_D)
            hist_loss[phase + '_perceptual'].append(cur_loss_perceptual)
            print('Epoch {:d}/{:d}  {:s}  loss_pxl {:.4f}  loss_adv {:.4f}  loss_D {:.4f}  loss_perceptual {:.4f}'.format(
                  epoch + 1, n_epochs, phase, cur_loss_pxl, cur_loss_adv, cur_loss_D, cur_loss_perceptual))

        print()

    print('Done!')
    return hist_loss


def main():
    parser = argparse.ArgumentParser(description='Image Outpainting GAN Training')
    parser.add_argument('--train_dir', type=str, default='coco128_final/train', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='coco128_final/validation', help='Path to validation data')
    parser.add_argument('--test_dir', type=str, default='coco128_final/test', help='Path to test data')
    parser.add_argument('--model_save_path', type=str, default='outpaint_models', help='Path to save models')
    parser.add_argument('--html_save_path', type=str, default='outpaint_html', help='Path to save HTML visualizations')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=251, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--outpaint', action='store_true', default=True, help='Outpainting mode (vs inpainting)')
    parser.add_argument('--input_size', type=int, default=128, help='Input image size')
    parser.add_argument('--output_size', type=int, default=192, help='Output image size')
    parser.add_argument('--extra_upsample', action='store_true', default=True, help='Add extra upsampling in generator')
    parser.add_argument('--adv_weight', type=float, default=0.001, help='Weight for adversarial loss')
    parser.add_argument('--save_interval', type=int, default=10, help='Interval for saving models and generating visualizations')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch for resumed training')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    args = parser.parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Create directories if they don't exist
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.exists(args.html_save_path):
        os.makedirs(args.html_save_path)

    # Load data
    data_loaders = get_data_loaders(args.train_dir, args.val_dir, args.test_dir, 
                                    batch_size=args.batch_size, 
                                    input_size=args.input_size, 
                                    output_size=args.output_size,
                                    outpaint=args.outpaint,
                                    num_workers=0)  # Set to 0 to avoid multiprocessing issues

    # Initialize models
    G_net = CEGenerator().to(device)
    D_net = CEDiscriminator().to(device)

    # Initialize weights
    if not args.resume:
        G_net.apply(weights_init_normal)
        D_net.apply(weights_init_normal)
    else:
        # Load pretrained models
        G_net.load_state_dict(torch.load(os.path.join(args.model_save_path, f'G_{args.start_epoch}.pt')))
        D_net.load_state_dict(torch.load(os.path.join(args.model_save_path, f'D_{args.start_epoch}.pt')))
        print(f'Loaded models from epoch {args.start_epoch}')

    # Loss functions
    criterion_pxl = nn.L1Loss()  # Pixel-wise loss
    criterion_D = nn.MSELoss()   # Adversarial loss

    # Optimizers
    optimizer_G = optim.Adam(G_net.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(D_net.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Check if we have valid data loaders
    if data_loaders.get('train') is None:
        raise ValueError("Training dataset is empty or directory doesn't exist. Please check your train_dir path.")
    
    # For validation, we can continue without it, but let's inform the user
    if data_loaders.get('val') is None:
        print("Warning: No validation dataset available. Training will proceed without validation.")
        # Create a dummy empty phase for 'val' to avoid errors in train_CE
        data_loaders['val'] = []
    
    # Train model
    hist_loss = train_CE(G_net, D_net, device, criterion_pxl, criterion_D, optimizer_G, optimizer_D,
                         data_loaders, args.model_save_path, args.html_save_path, n_epochs=args.epochs,
                         start_epoch=args.start_epoch, outpaint=args.outpaint, 
                         adv_weight=args.adv_weight, save_interval=args.save_interval)

    # Save loss history
    with open(os.path.join(args.model_save_path, 'loss_history.pkl'), 'wb') as f:
        pickle.dump(hist_loss, f)


if __name__ == '__main__':
    main()