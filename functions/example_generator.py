import torch
import os
from torchvision.utils import save_image
from progressive_gan import Generator
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation, PillowWriter
from IPython.display import HTML
from math import log2
import numpy as np
import imageio


def generate_examples(gen, steps, z_dim, device='cpu', n=100, path='saved_examples', epoch=None):
    folder = f'step{steps}'
    if epoch is not None:
        folder = f'step{steps}_epoch{epoch}'

    gen.eval()
    alpha = 1.0
    for i in range(n):
        with torch.no_grad():
            noise = torch.randn(1, z_dim, 1, 1).to(device)
            img = gen(noise, alpha, steps)
            if not os.path.exists(os.path.join(path, f'{folder}')):
                os.makedirs(os.path.join(path, f'{folder}'))
            save_image(img * 0.5 + 0.5, os.path.join(path, f'{folder}/img_{i}.png'))
    gen.train()


def _generater_sorter(x):
    padding = 7
    key = x.split('_')[1].split('.')[0]
    if len(x.split('_')) < 3:
        # print(int(key+padding*'9'))
        return int(key + padding * '9')  # one is required otherwise 0 is not considered when converting to int
    else:
        key2 = x.split('_')[2].split('.')[0]
        # print(int(key+key2+(padding-len(key2))*'9'))  
        return int(key + (padding - len(
            key2)) * '0' + key2)  # one is required otherwise 0 is not considered when converting to int


def _pad_images(images, target_size=256):
    _, _, h, w = images.shape
    pad_h = max(target_size - h, 0)
    pad_w = max(target_size - w, 0)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    padded_images = torch.nn.functional.pad(images, padding, value=1)
    return padded_images / 2 + 0.5  # Normalize to [0, 1]


def _create_grid(images, target_size, ncols=5):
    batch_size, channels, _, _ = images.shape
    nrows = (batch_size + ncols - 1) // ncols
    padded_images = _pad_images(images, target_size)
    grid = padded_images.new_zeros((channels, target_size * nrows, target_size * ncols))
    for i in range(batch_size):
        row = i // ncols
        col = i % ncols
        start_y = row * target_size
        start_x = col * target_size
        grid[:, start_y:start_y + target_size, start_x:start_x + target_size] = padded_images[i]
    return grid


def generate_example_animation(path, z_dim, in_channels, channels_img, fps=3, device='cpu', display_animation=False):
    generators = []
    for file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, file)):
            continue
        if not file.endswith('.pth'):
            continue
        if file.startswith('generator'):
            generators.append(file)
            print(os.path.join(path, file))
        # os.system(f'ffmpeg -r 10 -i {os.path.join(path, dir)}/img_%d.png -vcodec mpeg4 -y {os.path.join(path, dir)}.mp4')
        # os.system(f'rm -r {os.path.join(path, dir)}')

    generators = sorted(generators, key=_generater_sorter)
    print(generators)

    gen = Generator(
        z_dim, in_channels, img_channels=channels_img
    ).to(device)

    img_list = []
    grid_size = (5, 5)
    amount_gens = grid_size[0] * grid_size[1]
    fig = plt.figure(figsize=(8, 8))
    plt.axis('off')
    ims = []
    alpha = 1.0
    sub_path = 'evolution_grids'

    # noises = []
    # for i in range(25):
    #     noises.append(torch.randn(1, z_dim, 1, 1))
    fixed_noise = torch.randn(amount_gens, z_dim, 1, 1).to(device)

    max_size = int(generators[-1].split('_')[1].split('.')[0])
    print(max_size)

    # Create a list to store frames
    frames = []

    for gen_filename in generators:
        file_id = '_'.join(gen_filename.split('_')[1:]).split('.')[0]
        if len(file_id.split('_')) < 2:
            file_id += '_last_epoch'

        image_size = int(gen_filename.split('_')[1].split('.')[0])
        padding = (max_size - image_size) // 2
        step = int(log2(int(image_size) / 4))

        gen.load_state_dict(torch.load(os.path.join(path, gen_filename)))
        with torch.no_grad():
            fake = gen(fixed_noise, alpha, step).detach().to(device)
        # img = vutils.make_grid(fake, padding=padding, normalize=True, nrow=grid_size[0], pad_value=1.0)

        grid = _create_grid(fake, max_size, ncols=5)
        # print(grid)

        # img_list.append(img)
        img_list.append(grid)

        os.makedirs(os.path.join(path, sub_path), exist_ok=True)

        # save_image(img, os.path.join(path, sub_path, f'evolution_grid_{file_id}.png'))
        # ims = ([plt.imshow(np.transpose(img,(1,2,0)), animated=True)]) #TODO: maybe add epoch & size description?
        save_image(grid, os.path.join(path, sub_path, f'evolution_grid_{file_id}.png'))
        # ims.append([plt.imshow(np.transpose(grid,(1,2,0)), animated=True)]) #TODO: maybe add epoch & size description?

        grid_np = grid.permute(1, 2, 0).cpu().numpy() * 255  # Convert to numpy and scale to [0, 255]
        grid_np = np.uint8(grid_np)  # Convert to uint8 for imageio

        frames.append(grid_np)  # Append the first frame

        # # Assuming you want to create a GIF with 10 frames
        # for _ in range(9):
        #     # Add more frames here if needed
        #     frames.append(grid_np)  # Append the same frame multiple times for demonstration

    # ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
    # ani = ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    # ani.save(os.path.join(path, sub_path,'evolution.gif'), dpi=300, writer=PillowWriter(fps=5))

    # if display_animation:
    #     HTML(ani.to_jshtml())

    # Save frames as a GIF
    imageio.mimsave(os.path.join(path, sub_path, 'evolution.gif'), frames,
                    fps=fps)  # duration=1.5)  # Adjust duration as needed

    # return ani


if __name__ == '__main__':
    generate_example_animation('outputs/32_gens_of_runs_23_and_31_combined', 256, 256, 3, fps=3, display_animation=True)
