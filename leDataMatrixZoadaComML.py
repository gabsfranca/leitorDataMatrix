import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

img_size = (150, 150)
num_classes = 9  

class SimpleCnn(nn.Module):
    def __init__(self):
        super(SimpleCnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * (img_size[0] // 4) * (img_size[1] // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * (img_size[0] // 4) * (img_size[1] // 4))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCnn()
model.load_state_dict(torch.load('modelo_datamatrix.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

image_paths = ["zoada_111_18.png\\111_18.png_0.png", "zoada_111_18.png\\111_18.png_1.png", "zoada_111_18.png\\111_18.png_2.png", "zoada_111_18.png\\111_18.png_3.png", "zoada_111_18.png\\111_18.png_4.png", "zoada_111_18.png\\111_18.png_5.png", "zoada_111_18.png\\111_18.png_6.png", "zoada_111_18.png\\111_18.png_7.png", "zoada_111_18.png\\111_18.png_8.png", "zoada_111_18.png\\111_18.png_9.png", "zoada_111_18.png\\111_18.png_10.png", "zoada_111_18.png\\111_18.png_11.png", "zoada_111_18.png\\111_18.png_12.png", "zoada_111_18.png\\111_18.png_13.png", "zoada_111_18.png\\111_18.png_14.png", "zoada_111_18.png\\111_18.png_15.png", "zoada_111_18.png\\111_18.png_16.png", "zoada_111_18.png\\111_18.png_17.png", "zoada_111_18.png\\111_18.png_18.png", "zoada_111_18.png\\111_18.png_19.png"]
class_names = ['111', '222', '333', '444', '555', '666', '777', '888', '999']
for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)


    predicted_class = class_names[predicted.item()]

    print(f'Predicted Class: {predicted_class}')
