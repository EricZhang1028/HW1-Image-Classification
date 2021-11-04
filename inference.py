import os
import numpy as np
import torch
import torchvision
from PIL import Image

def get_classes():
    classes = []
    with open('./classes.txt', 'r') as f:
        classes = [x.strip() for x in f.readlines()]

    return classes


# define transform
img_size = 224
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(img_size),
    torchvision.transforms.CenterCrop(img_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

# load model and classes
model = torch.load('model.pt')
classes = get_classes()

# start to predict test data
print("Start to generate answer.txt file...")
model.eval()
test_dir = './dataset/testing_images/'
with open('testing_img_order.txt') as f:
    test_images = [x.strip() for x in f.readlines()]  # all the testing images
    
    submission = []
    for img in test_images:  # image order is important to your result
        img_path = test_dir + img
        img_PIL = Image.open(img_path)
        img_tensor = transforms(img_PIL)
        img_tensor.unsqueeze_(0)
        data = img_tensor.cuda()
        
        out = model(data)
        _, predicted = torch.max(out.data, 1)
        pre_class = classes[predicted]
        
        submission.append([img, pre_class])

# save result as an answer file
np.savetxt('answer.txt', submission, fmt='%s')
print("answer.txt file is generated.")