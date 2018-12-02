import pickle

import tensorboardX
import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms

from dataset_handler.colorisation_dataset import ColorizationDataset
from nn_utils.model import UNet, EncoderNet, AlexNet_finetune
from nn_utils.losses import AdversarialConditionalLoss, AdversarialLoss

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_transform = transforms.Compose(
                      [transforms.Resize((224, 224)), 
                       #transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
                       torchvision.transforms.RandomAffine(5, [0.1, 0.1], [0.95, 1.05]),
                       transforms.ToTensor()])
path = "/data/places365standard_easyformat/places365_standard/train"

my_dataset = ColorizationDataset(path, transform=train_transform)
train_data = torch.utils.data.DataLoader(my_dataset, batch_size=32, shuffle=True, num_workers=4)
gen = UNet([3, 20, 40, 80]).to(device=device)
#disc = EncoderNet([6, 40, 80, 160, 320]).to(device=device)
disc = AlexNet_finetune().to(device=device)


print(gen, disc)
adv_loss = AdversarialLoss(gen, disc, device)

gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
disc_optimizer = torch.optim.Adam(disc.linear.parameters(), lr=0.001, betas=(0.5, 0.999))

writer = tensorboardX.SummaryWriter(log_dir="./logs", comment="pix2pix")
for e in range(100):
    for i, (x, y) in enumerate(train_data):
        print(i)
        disc_loss = adv_loss.discriminator_forward(x, y, None, verbose=i % 100 == 0)
        disc_loss.backward()
        disc_optimizer.step()
       
        gen_loss = adv_loss.generator_forward(x, y, None, verbose=i % 100 == 0)
        gen_loss.backward()
        gen_optimizer.step()

        if i % 100 == 0:
            writer.add_scalars("pix2pix/", {"Discriminator": disc_loss, "Generator loss": gen_loss}, e * len(train_data) + i)


            print("Discriminator loss : ", disc_loss)
            print("Generator loss : ", gen_loss)
            viz = vutils.make_grid(torch.cat([y.to(device), gen(x.to(device))], dim=0))
            viz = torch.clamp(viz, 0, 0.999999)
            writer.add_image('visu/', viz, e * len(train_data) + i)
            pickle.dump(gen, open("generator_unet_{}.pkl".format(e), "wb"))




