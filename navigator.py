# Latent Space Navigator
import numpy as np
from audio import AudioHandler
import torch

class LatentNavigator(object):

    def __init__(self, Audio: AudioHandler, latent_vectors):
        self.latent_vectors = latent_vectors
        self.latent_size = latent_vectors[0].shape
        self.frame = 0
        # self.max = np.array([0.000001 for i in range(latent_dims)])
        self.location = latent_vectors[0]
        self.step_size = 0.1
        self.Audio = Audio
    
    # audio has shape (num_channel, stream_len)
    def navigate(self):
        self.frame += 1
        a = min(self.Audio.extract_rms() * 3, 1)
        r = torch.randn((3, 32, 32), dtype=torch.float32, device="cpu")
        print(a)
        return self.latent_vectors[0] * (1-a) + r * a
        return self.latent_vectors[self.frame % len(self.latent_vectors)]

        # rms = Audio.extract_rms()
        # centroid = Audio.extract_centroid()
        fft = self.Audio.extract_fft()
        # freqs = Audio.extract_fft_freqs()

        fft = np.sum(fft, axis=0)

        split = np.array_split(fft, self.latent_dims)
        split = list(map(lambda x : abs(np.average(x)), split))
        self.max = np.maximum(split, self.max)

        # split = list(map(lambda x, i: x * 3/self.max[i], split))

        offset = [((split[i] / self.max[i]) ** 1.5) for i in range(len(split))] # normalized offset vector shape (latent_dims)

        self.max *= 0.99

        displacement = self.target - self.location
        disp_len = np.sqrt(displacement.dot(displacement))
        
        self.location = displacement / disp_len * self.step_size * offset[0] + self.location

        if disp_len < 0.1:
            self.target = np.random.normal(np.zeros(self.latent_dims))
            # print("new location")

        # print(self.location, offset)

        return self.location + offset