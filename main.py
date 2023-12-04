from navigator import LatentNavigator
import matplotlib.pyplot as plt
import K
import numpy as np
import torch.distributions
import torch.utils
import torch.nn.functional as F
import torch.nn as nn
import torch
import argparse
import librosa
# from utils import Timer
import time
import sounddevice as sd
from multiprocessing import Process,Queue,Pipe
from audio import AudioHandler
import pygame
import pickle
import enum
plt.rcParams['figure.dpi'] = 200
import sys
from eval import loadModel, get_image, get_latent_vectors

def run_visuals(input_device, channels):

    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode((600, 600), pygame.RESIZABLE)
    run_loop = True
    Audio = AudioHandler(input_device=input_device, channels=channels)
    Audio.start()
    # timer = Timer(use_hertz=False)
    device = 'cuda' if torch.cuda.is_available() and len(K.gpu_ids) > 0 else 'cpu'
    net = loadModel(device=device)
    latent_vectors = get_latent_vectors(filepaths=K.filepaths, net=net, device=device)
    Navigator = LatentNavigator(Audio=Audio, latent_vectors=latent_vectors)

    try:
        while run_loop:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run_loop = False

            # while not visualizer.event_queue.empty():
            #     pickled_event = visualizer.event_queue.get()
            #     event = pickle.loads(pickled_event)
            #     if type(event) == SetConfig:
            #         set_config = event
            #         # TODO: this resets effect state, keep old effects and also add new ones
            #         effect_objects = [EffectChooser().create_effect(effect_config) for effect_config in set_config.get_effects() if effect_config.active]
            #         Audio.update_set_config(event)


            # Audio.update_set_config(new_set_config) # if we wanted to change parameters during
            Audio.listen()
            latent_location = Navigator.navigate()
            # latent_location = torch.randn((3, 32, 32), dtype=torch.float32, device="cpu")
            image = get_image(net=net, z=latent_location)
            surf = pygame.surfarray.make_surface(image)

            # print("fps: ", 1 / (time.time() - t))

            # t = time.time()
            # Scale the image to your needed size
            # surf = apply_effects(effect_objects, surf)
            surf = pygame.transform.scale(surf, K.default_image_size)

            screen_width, screen_height = window.get_size()
            content_width, content_height = surf.get_size()
            content_x = (screen_width - content_width) // 2
            content_y = 0

            # Create a numpy array from the content image
            content_array = pygame.surfarray.array3d(surf)
            # average_color = np.mean([content_array[:5, :, :], content_array[-5:, :, :]], axis=(0, 1, 2)).astype(np.uint8)
            average_color = np.mean(content_array, axis=(0, 1)).astype(np.uint8)
            # Calculate the average color of the existing content
            # average_color = np.mean(content_array, axis=(0, 1)).astype(np.uint8)
            window.fill(average_color)
            window.blit(surf, (content_x, 0))
            pygame.display.update()
            window.fill((0, 0, 0))
            # timer.lap("end")
    except (KeyboardInterrupt, SystemExit):
        print('Keyboard interrupt detected. Exiting...')
        pygame.quit()
        Audio.stop()
        raise
    pygame.quit()
    Audio.stop()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '-l', '--list-devices', action='store_true',
        help='show list of audio devices and exit')
    args, remaining = parser.parse_known_args()
    if args.list_devices:
        print(sd.query_devices())
        parser.exit(0)
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parser])
    parser.add_argument(
        '-i', '--input-device', type=int,
        help='input device (numeric ID or substring)')
    parser.add_argument(
        '-o', '--output-device', type=int,
        help='output device (numeric ID or substring)')
    parser.add_argument(
        '-c', '--channels', type=int, default=1,
        help='number of channels')
    args = parser.parse_args(remaining)
    run_visuals(args.input_device, args.channels)