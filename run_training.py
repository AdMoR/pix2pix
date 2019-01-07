import pickle
import os

import tensorboardX
from skimage import color
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms

from dataset_handler.colorisation_dataset import ColorizationDataset
from dataset_handler.double_image_dataset import DoubleImageDataset
from dataset_handler.ade20k_loader import EdgeADE20k, NormalADE20k
from dataset_handler.edge2img_dataset import EdgesDataset
from nn_utils.model import UNet, EncoderNet
from nn_utils.colorisation_model import ColorUNet
from nn_utils.gan_losses import AdversarialConditionalLoss, AdversarialLoss
from nn_utils.combined_loss import CombinedGANLoss

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_transform = transforms.Compose(
                      [#transforms.Resize((256, 256)),
                       transforms.RandomResizedCrop(512),
                       transforms.RandomHorizontalFlip(),
                       torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
                       #torchvision.transforms.RandomAffine(5, [0.1, 0.1], [0.95, 1.05]),
                       transforms.ToTensor()])
path = "/data/ADE20K_2016_07_26/"

my_dataset = NormalADE20k(path, is_transform=True)
#my_dataset = ColorizationDataset(path, transform=train_transform)
#my_dataset = EdgesDataset(path, transform=train_transform)
#my_dataset = DoubleImageDataset(path, transform=train_transform)
train_data = torch.utils.data.DataLoader(my_dataset, batch_size=2, shuffle=True, num_workers=2)
#gen = ColorUNet().to(device=device)
gen = UNet([151, 64, 128, 256, 256, 512, 512, 512, 512, 512], target_dim=3).to(device=device)
#disc = EncoderNet([4, 64, 128, 256, 512, 512, 512]).to(device=device)

print(gen)
adv_loss = CombinedGANLoss([
    AdversarialConditionalLoss(gen, EncoderNet([154, 64, 128, 256, 512, 512, 512]).to(device=device), device=device),
    AdversarialConditionalLoss(gen, EncoderNet([154, 64, 128, 256, 256, 512, 512, 512]).to(device=device), device=device),
], device)

gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(adv_loss.disc_parameters(), lr=0.0002, betas=(0.5, 0.999))


writer = tensorboardX.SummaryWriter(log_dir="./logs2", comment="pix2pix")
for e in range(10000):
    for i, (x, y) in enumerate(train_data):
        disc_optimizer.zero_grad()
        disc_loss = adv_loss.discriminator_forward(x, y, None)
        disc_loss.backward()
        disc_optimizer.step()

        gen_optimizer.zero_grad()
        gen_loss = adv_loss.generator_forward(x, y, None)
        gen_loss.backward()
        gen_optimizer.step()

        if i % 100 == 0:
            writer.add_scalars("pix2pix/", {"Discriminator": disc_loss, "Generator loss": gen_loss}, e * len(train_data) + i)
            my_dataset.build_visu(writer, gen, x, y, device, e * len(train_data) + i)

    if e % 10 == 0:
        pickle.dump(gen, open("generator_unet_{}_{}.pkl".format(os.path.basename(path), e), "wb"))




