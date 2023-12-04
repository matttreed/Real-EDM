import math
import argparse
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import util
import time
import K
import numpy as np
from PIL import Image
from util.interpolation import spherical_blend

from models import RealNVP, RealNVPLoss
from tqdm import tqdm


def loadModel(device):
    net = RealNVP(
        num_scales=K.num_scales, 
        in_channels=K.in_channels, 
        mid_channels=K.mid_channels, 
        num_blocks=K.num_blocks
        )
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, K.gpu_ids)
        # cudnn.benchmark = args.benchmark

    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('ckpts/best.pth.tar')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    return net

def get_image(net, z):
    with torch.no_grad():
        x, _ = net(torch.unsqueeze(z,dim=0), reverse=True)
        image = torch.sigmoid(x).squeeze(dim=0).numpy()
        image = image.transpose(1, 2, 0)
        image = np.rot90(image, k=1)
        image = np.flipud(image)
        image = (image * 255).astype(np.uint8)
        return image


def load_and_transform_image(image_path):
    image = Image.open(image_path).convert('RGB')

    transformation = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    image_tensor = transformation(image)

    return image_tensor

def get_latent_vectors(filepaths, net, device):
    vectors = []
    for filepath in filepaths:
        x = load_and_transform_image(filepath)
        x = x.unsqueeze(0)
        x = x.to(device)
        z, _ = net(x, reverse=False)
        vectors.append(z.squeeze(0))
    return vectors

def getZ(net, image):
    return net(image, reverse=False)


def main(args):
    device = 'cuda' if torch.cuda.is_available() and len(args.gpu_ids) > 0 else 'cpu'

    print('Building model..')
    net = RealNVP(num_scales=2, in_channels=3, mid_channels=64, num_blocks=8)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, args.gpu_ids)
        cudnn.benchmark = args.benchmark

    # Load checkpoint.
    print('Resuming from checkpoint at ckpts/best.pth.tar...')
    assert os.path.isdir('ckpts'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('ckpts/best.pth.tar')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    # global best_loss
    # best_loss = checkpoint['test_loss']
    # start_epoch = checkpoint['epoch']

    with torch.no_grad():
        num_samples = 64

        # images = sample(net, num_samples, device)
        # os.makedirs('samples', exist_ok=True)
        # images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        # torchvision.utils.save_image(images_concat, 'samples/test.png')

        x1 = load_and_transform_image("data/test_images/5.png")
        x1 = x1.unsqueeze(0)
        x1 = x1.to(device)
        x2 = load_and_transform_image("data/test_images/4.png")
        x2 = x2.unsqueeze(0)
        x2 = x2.to(device)
        x3 = load_and_transform_image("data/test_images/1.png")
        x3 = x3.unsqueeze(0)
        x3 = x3.to(device)
        x4 = load_and_transform_image("data/test_images/3.png")
        x4 = x4.unsqueeze(0)
        x4 = x4.to(device)
        # z, sldj = net(x, reverse=False)
        # xp, _ = net(z, reverse=True)
        # xp = torch.sigmoid(xp)

        z1, sldj = net(x1, reverse=False)
        z2, sldj = net(x2, reverse=False)
        z3, sldj = net(x3, reverse=False)
        z4, sldj = net(x4, reverse=False)

        inter_z = torch.empty(64, 3, 32, 32)

        for i in range(8):
            for j in range(8):
                weights = [
                    (i / 7) * (j / 7),
                    (i / 7) * (1 - (j/7)),
                    (1 - (i/7)) * (j / 7), 
                    (1 - (i/7)) * (1 - (j / 7))
                ]
                total = sum(weights)
                inter_z[i * 8 + j] += z1.squeeze(0) * weights[0] / total
                inter_z[i * 8 + j] += z2.squeeze(0) * weights[1] / total
                inter_z[i * 8 + j] += z3.squeeze(0) * weights[2] / total
                inter_z[i * 8 + j] += z4.squeeze(0) * weights[3] / total

                # a = i / 63
                # inter_z[i] = z1 * a + z2 * (1-a)

        # Loop to create copies with added Gaussian noise
        # for i in range(64):
        #     # Copy the original tensor
        #     theta = np.pi / 2 * (i / 63)
        #     inter_z[i] = z1 * np.sin(theta) + z2 * np.cos(theta)
        #     # a = i / 63
        #     # inter_z[i] = z1 * a + z2 * (1-a)

        xp, _ = net(inter_z, reverse=True)
        xp = torch.sigmoid(xp)

        # xp -= x
        images_concat = torchvision.utils.make_grid(xp, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        torchvision.utils.save_image(images_concat, 'samples/test2.png')





        # z, sldj = net(x, reverse=False)


        # noisy_z = torch.empty(64, 3, 32, 32)

        # # Loop to create copies with added Gaussian noise
        # for i in range(64):
        #     # Copy the original tensor
        #     noisy_z[i] = z.clone()

        #     # Generate Gaussian noise with increasing standard deviation
        #     noise = torch.randn_like(z) * (i / 63)

        #     # Add the noise to the copied tensor
        #     noisy_z[i] += noise.squeeze(0)

        # xp, _ = net(noisy_z, reverse=True)
        # xp = torch.sigmoid(xp)

        # # xp -= x
        # images_concat = torchvision.utils.make_grid(xp, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        # torchvision.utils.save_image(images_concat, 'samples/test.png')
        # print(x)
        # print(xp)

        # print(x)

        # z, sldj = net(x, reverse=False)

        # z1 = getZ(image)
        # z2 = getZ(image2)

        # start_time = time.time()

        # # Your for loop here
        # for i in tqdm(range(500)):
        #     image = sample(net, num_samples, device)

        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")

        # theta = 0

        return

        while True:
            # z_int = z * math.sin(theta) + z2 * math.cos(theta)

            # img = net(z_int, reverse=True)

            theta += .01

def sample(net, batch_size, device):
    """Sample from RealNVP model.

    Args:
        net (torch.nn.DataParallel): The RealNVP model wrapped in DataParallel.
        batch_size (int): Number of samples to generate.
        device (torch.device): Device to use.
    """
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    x, _ = net(z, reverse=True)
    x = torch.sigmoid(x)

    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RealNVP on CIFAR-10')

    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--benchmark', action='store_true', help='Turn on CUDNN benchmarking')
    parser.add_argument('--gpu_ids', default='[0]', type=eval, help='IDs of GPUs to use')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--max_grad_norm', type=float, default=100., help='Max gradient norm for clipping')
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of epochs to train')
    parser.add_argument('--num_samples', default=64, type=int, help='Number of samples at test time')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of data loader threads')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--weight_decay', default=5e-5, type=float,
                        help='L2 regularization (only applied to the weight norm scale factors)')

    best_loss = 0

    main(parser.parse_args())