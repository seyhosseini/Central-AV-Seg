print("Importing ...")
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.utils.data import DataLoader
import torchvision.transforms as transforms # Using TorchIO may help in 3D augmentation *
import nibabel as nib
import numpy as np
import random

# Define your model architecture here
print("Defining Classes ...")

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels , out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module): #
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2] # NCXYZ
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2))
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module): ### Add dropout!
    def __init__(self, in_channels, out_channels):
        super(UNet3D, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output


# Define a custom transform class for applying the same random crop
class RandomCrop3D: ###
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        inputs, targets = sample

        # Get the input size
        input_size = inputs.shape[2:] ###

        # Calculate the starting index for the crop
        start_indexes = [random.randint(0, input_size[i] - self.output_size[i]) for i in range(3)]

        # Perform the crop on both inputs and targets
        inputs  = inputs [:,:, start_indexes[0]:start_indexes[0] + self.output_size[0], 
                               start_indexes[1]:start_indexes[1] + self.output_size[1],
                               start_indexes[2]:start_indexes[2] + self.output_size[2]]

        targets = targets[:,:, start_indexes[0]:start_indexes[0] + self.output_size[0], 
                               start_indexes[1]:start_indexes[1] + self.output_size[1],
                               start_indexes[2]:start_indexes[2] + self.output_size[2]]

        return inputs, targets

# Define the output size for random cropping
output_size = (128, 128, 128)

# Define the transforms
transform = transforms.Compose([
    RandomCrop3D(output_size),              # Custom random crop
    # transforms.RandomVerticalFlip(),        # Random vertical flipping
    # transforms.RandomHorizontalFlip()        # Random horizontal flipping
])


# Define your dataset class for loading CT images and masks

class CTImageDataset(torch.utils.data.Dataset): ###
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths

    def __getitem__(self, index):
        image = nib.load(self.image_paths[index]).get_fdata()
        mask  = nib.load(self.mask_paths [index]).get_fdata()
        image = torch.from_numpy(image) .unsqueeze(0).float() ### 1-Channel?!
        mask  = torch.from_numpy(mask ) .unsqueeze(0).long() ### Changed!
        return image, mask

    def __len__(self):
        return len(self.image_paths)

# Define your training function

def train(model, train_loader, criterion, optimizer, device): ###
    model.train() ###
    running_loss = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader):
        # print(f"Batch {batch_idx+1} Started")

        images = images.to(device)
        masks  = masks .to(device)

        # Apply transforms to the inputs and targets
        images, masks = transform((images, masks))

        optimizer.zero_grad()

        # Forward pass
        # print("Passing through Model ...")
        outputs = model(images)

        # Compute loss
        # print("CrossEnthropy() ...")
        loss = criterion(outputs, torch.squeeze(masks, dim=1)) ###

        # Backward pass and optimization
        # print("Backward ...")
        loss.backward()
        # print("Step ...")
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)






# Set your training parameters
print("Setting Parameters & Instanciating ...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") ####1
epochs = 10
batch_size = 1 #4 ###
learning_rate = 0.0001 #0.001 ###

# Create your model instance

model = UNet3D(in_channels=1, out_channels=3)
model = model.to(device)

# Create your dataset and data loader instances

image_paths_train = ["Data\SPIROMCS-Case36-Vx3.nii.gz", "Data\SPIROMCS-Case43-Vx3.nii.gz"]
mask_paths_train  = ["Data\SPIROMCS-Case36-012Labelmap.nii.gz", "Data\SPIROMCS-Case43-012Labelmap.nii.gz"]
train_dataset = CTImageDataset(image_paths_train, mask_paths_train) ### Cases 43&36 ### M:1 A:2 V:3 > 012!
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) ### Mask: B=1?C=1?XYZ? #shuffle=True

# Define your loss function and optimizer

criterion = nn.CrossEntropyLoss() ####2 ignore_index (int, optional) ***
optimizer = optim.Adam(model.parameters(), lr=learning_rate) ###

# Start the training loop
print("Start Training ...")

for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device) ########
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

# Save the trained model

torch.save(model.state_dict(), "model.pth") ###


# model.eval()
# for images, masks in train_loader:
# nn.CrossEntropyLoss(): label_smoothing=0.0?!!
