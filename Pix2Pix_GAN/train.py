import torch 
#
import torch.nn as nn
import torch.optim as optim
import config
from utils import save_checkpoint, load_checkpoint, save_some_examples
from tqdm import tqdm
from discriminator import Discriminator
from generator import Generator
from torch.utils.data import DataLoader
from dataset import MapDataset
from torchvision.utils import save_image

torch.backends.cudnn.benchmark = True

def train_fn(
        disc,gen,loader,opt_disc,opt_gen,l1_loss,bce,g_scaler,d_scaler,
):
    loop = tqdm(loader,leave=True)

    for idx,(x,y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        #Train Discriminator 
        #Automatic mixed precision is a technique used to accelerate deep learning models by performing some computations in lower precision (e.g., half-precision or FP16) while maintaining the accuracy benefits of higher precision (e.g., single-precision or FP32). This technique takes advantage of the fact that some operations, such as matrix multiplications, can be safely performed in lower precision without significantly affecting the overall accuracy of the model.
        with torch.cuda.apm.autocast():
            y_fake = gen(x)  #generate fake image
            D_real = disc(x,y)  #give real and transformed label image to disc
            D_real_loss = bce(D_real,torch.ones_like(D_real)) #label should be one bacause it is real image and calculate the loss
            D_fake = disc(x,y_fake.detach()) #give real image and generated fake image of generator to disc
            D_fake_loss = bce(D_fake,torch.zero_like(D_fake))  #label should be zero as fake image is going and disc should know to train
            D_loss = (D_real_loss+D_fake_loss)/2
        """
        Zeroes out the gradients of the discriminator.
        Scales the loss, performs the backward pass, and computes the gradients.
        Updates the discriminator's parameters using the optimizer and the computed gradients.
        Updates the scaling factor to maintain numerical stability for subsequent iterations.
        """
        disc.zero_grad()  #clear the gradient accumulation
        d_scaler.scale(D_loss).backward()  #backward calculate gradients . function applies scaling to the loss value, which is necessary for automatic mixed precision training.
        d_scaler.step(opt_disc)  #change the weights 
        d_scaler.update() #This step is specific to torch.cuda.amp.GradScaler and is used to update the scaling factor. The scaler adjusts the scaling factor based on the gradients' norm to maintain numerical stability during training.


        #Train Generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x,y_fake)  #given image to disc
            G_fake_loss = bce(D_fake,torch.ones_like(D_fake))  #it should predict it is real image
            L1 = l1_loss(y_fake,y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1 
            #L1 loss - because we dont want you generate image should differ from the real label so 
            #now loss should focus on two things first trying to fool and second trying to generate as same as label image

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step()
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real = torch.sigmoid(D_real).mean().item(),
                D_fake = torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3,features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(),lr = config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen = optim.Adam(gen.parameters(),lr = config.LEARNING,betas=(0.5,0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,gen,opt_gen,config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC,disc,opt_disc,config.LEARNING_RATE)
        
        train_dataset = MapDataset(root_dir=config.TRAIN_DIR)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS
        )
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        val_dataset = MapDataset(root_dir=config.VAL_DIR)
        val_loader = DataLoader(val_dataset,batch_size=1,shuffle=False)

        for epoch in range(config.NUM_EPOCHS):
            train_fn(
                disc,gen,train_loader,opt_disc,opt_gen,L1_LOSS,BCE,g_scaler,d_scaler,
            )
            if config.SAVE_MODEL and epoch % 5 == 0:
                save_checkpoint(gen,opt_gen,filename=config.CHECKPOINT_GEN)
                save_checkpoint(disc,opt_disc,filename=config.CHECKPOINT_DISC)


if __name__ == "__main__":
    main()