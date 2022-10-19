import gradio as gr
import torch
from torch import nn
import os
from torchvision.utils import save_image
LATENT_SIZE=50

# задаем класс модели
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(LATENT_SIZE+10, 512, kernel_size=2, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(128, 28, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.ReLU(True)
        )
        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(28, 1, kernel_size=2, stride=2, padding=2, bias=False),
            nn.Tanh(),
        )
    def forward(self, x, class_vec):
        x = torch.cat([x,class_vec], 1)
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)
        return x

    
# функция генерации: на вход число картинок
# на выходе имя файла
def generate(number, c):
    global gen
    gen.eval()
    noise = torch.randn(number, 50, 1,1).to('cpu')
    c = torch.tensor([c])
    target = nn.functional.one_hot(c.unsqueeze(1).unsqueeze(1).to(torch.int64), 10).permute(0,3,1,2).float().repeat(number, 1, 1, 1)
    tensors = gen(noise, target)
    save_image(tensors, '1.jpg', normalize=True)
    return '1.jpg'
    
# инициализация модели: архитектура + веса
def init_model():
    global gen
    gen = Generator()
    gen.load_state_dict(torch.load('num_generator_ws40.pt', map_location=torch.device('cpu')))
    return gen
# запуск gradio
def run(share=False):
    gr.Interface(
        generate,
        inputs=[gr.inputs.Slider(label='Количество цифр', minimum=1, maximum=10, step=1, default=1),
        gr.inputs.Slider(label='Число', minimum=1, maximum=10, step=1, default=1)],
        outputs = "image",
    ).launch(share=False)
if __name__ == '__main__':
    init_model()
    run()