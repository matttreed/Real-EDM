# constants 

latent_dims = 2
path = "vae/models/model"
device = "cpu"
epochs = 1
default_image_size = (1000,1000)

num_scales=2
in_channels=3
mid_channels=64
num_blocks=8
gpu_ids=[0]
batch_size=64
lr=1e-3
max_grad_norm=100
num_epochs=100
num_samples=64
num_workers=8
weight_decay=5e-5
filepaths=[
    "data/test_images/1.png",
    "data/test_images/2.png",
    "data/test_images/3.png",
    "data/test_images/4.png",
    "data/test_images/5.png"
]