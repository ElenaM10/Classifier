import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models 
from collections import OrderedDict
from PIL import Image
import json
import numpy as np
import os

def get_input_args():
    
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description = 'flower classification')
    parser.add_argument('--image', type = str, default='./flowers/test/27/image_06887.jpg',
                                 help = 'sample image file')
    parser.add_argument('--gpu', action = "store", default='cuda',
                                 help = 'either CPU or GPU if available')
    parser.add_argument('--save_dir', help="Save model to folder", default='/home/workspace/ImageClassifier/vgg16', action='store')
    parser.add_argument('--cat_to_name', type = str, default='cat_to_name.json', 
                                 help=' flower names from the flower file')
    parser.add_argument('--topk', type = int, default = 5, 
                         help = 'top 5 classes')
    args = parser.parse_args()
    return args

def label_mapping(cat_to_name):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    print(cat_to_name)
    return cat_to_name

def load_checkpoint(filepath):
    device = torch.device("cuda" if torch.cuda.is_available()else "cpu")
    checkpoint = torch.load(os.path.join(save_dir, "checkpoint.pth"))
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
     # TODO: Process a PIL image for use in a PyTorch model
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    processing_img = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor()])
    img_size = Image.open(image)
    img_size = processing_img(img_size).float()
    np_img = np.array(img_size)
    transposed_img = np.transpose(np_img , (1,2,0))
    np_img = (transposed_img - mean)/std
    np_img = np.transpose(np_img , (2,0,1))
    return torch.from_numpy(np_img)

def predict(image_path, model, topk=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    #Loading img 
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze_(0)
    img = img.to(device)
    with torch.no_grad():
        logps = model.forward(img)
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(topk , dim = 1)
    topk_ps = top_ps.tolist()[0]
    classes =top_class.squeeze(0).tolist()
    class_to_idx = model.class_to_idx
    idx_to_class = {value: key for key , value in model.class_to_idx.items()}
    
    topk_class = [idx_to_class[i] for i in classes]
    return topk_ps , topk_class
   
def main():
    in_args = get_input_args()
    cat_to_name = in_args.cat_to_name
    checkpoint = in_args.save_dir
    image = in_args.image
    topk =in_args.topk
    model_check = load_checkpoint(checkpoint)
    cat_to_name = label_mapping(cat_to_name)
    process_image(image)
    names, accuracy = predict(image,model_check, topk)
    print("Flower Label: {}, Accuracy:{}".format(cat_to_name[names[0]], str(accuracy[0])))
    
if __name__ == "__main__":
    main()  
    
   