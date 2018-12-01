import torch
import torchvision
import torchvision.transforms as transforms

from dataset_handler.colorisation_dataset import ColorizationDataset
from nn_utils.model import UNet, EncoderNet
from nn_utils.losses import AdversarialConditionalLoss


#torch.set_default_tensor_type('torch.cuda.FloatTensor')

train_transform = transforms.Compose(
                      [transforms.Resize((224, 224)), 
                       #transforms.RandomResizedCrop(224),
                       transforms.RandomHorizontalFlip(),
                       torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
                       torchvision.transforms.RandomAffine(5, [0.1, 0.1], [0.95, 1.05]),
                       transforms.ToTensor()])
path = "/data/places365standard_easyformat/places365_standard/train"

my_dataset = ColorizationDataset(path, transform=train_transform)
train_data = torch.utils.data.DataLoader(my_dataset, batch_size=1, num_workers=4)
gen = UNet([3, 10, 20, 40])
disc = EncoderNet([6, 20, 40, 60])
adv_loss = AdversarialConditionalLoss(gen, disc)

gen_optimizer = torch.optim.SGD(gen.parameters(), lr=0.0001)
disc_optimizer = torch.optim.SGD(disc.parameters(), lr=0.0001)

for e in range(100):
    for i, (x, y) in enumerate(train_data):

        disc_loss = adv_loss.discriminator_forward(x, y, None)
        disc_loss.backward()
        disc_optimizer.step()
       
        gen_loss = adv_loss.generator_forward(x, y, None)
        gen_loss.backward()
        gen_optimizer.step()

        if i % len(train_data) / 10 == 0:
            print("Discriminator loss : ", disc_loss)
            print("Generator loss : ", gen_loss)




