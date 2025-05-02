import torch
import numpy as np
import skimage.transform
from models import load_model
from utils import construct_masked, blend_result


def perform_outpaint(gen_model, input_img, blend_width=8):
    '''
    Performs outpainting on a single color image with arbitrary dimensions.
    Returns: 192x192 unmodified output + upscaled & blended output.
    '''
    # Enable evaluation mode
    gen_model.eval()
    torch.set_grad_enabled(False)

    # Construct masked input
    masked_img = construct_masked(input_img)
    
    # Convert to torch
    masked_img = masked_img.transpose(2, 0, 1)
    masked_img = torch.tensor(masked_img[np.newaxis], dtype=torch.float)

    # Call generator
    output_img = gen_model(masked_img)

    # Convert to numpy
    output_img = output_img.cpu().numpy()
    output_img = output_img.squeeze().transpose(1, 2, 0)
    output_img = np.clip(output_img, 0, 1)

    # Blend images
    norm_input_img = input_img.copy().astype('float')
    if np.max(norm_input_img) > 1:
        norm_input_img /= 255
    blended_img, src_mask = blend_result(output_img, norm_input_img, blend_width)
    blended_img = np.clip(blended_img, 0, 1)

    return output_img, blended_img