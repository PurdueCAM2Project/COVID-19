# PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
# by Bolei Zhou, sep 2, 2017
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


"""
original author: 
    PlacesCNN to predict the scene category, attribute, and class activation map in a single pass
    by Bolei Zhou, sep 2, 2017

adaptor: 
    Vishnu Banna

Contact: 
    DM cam2 slack: 
        Vishnu Banna
    email: 
        vbanna@purdue.edu
"""

"""
TODO: documentation
"""
class SceneDetectionClass():
    def __init__(self):
        # load the model
        self.classes, self.labels_IO, self.labels_attribute, self.W_attribute = self.load_labels()

        self.W_attribute = torch.from_numpy(self.W_attribute).cuda()

        self.features_blobs = [] #clear me dumb ass
        self.model = self.load_model()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        #load parameters and softmax weights
        self.params = list(self.model.parameters())

        self.params[-2].data = self.params[-2].data.cpu()

        self.weight_softmax = self.params[-2].data.numpy()
        self.weight_softmax[self.weight_softmax<0] = 0

        self.params[-2].data = self.params[-2].data.cuda()
        self.weight_softmax = torch.from_numpy(self.weight_softmax).cuda()

        #images to run
        self.img = None
        self.img_name = None
        self.og_image = None
        return


    def load_labels(self):
        # prepare all the labels
        # scene category relevant
        file_name_category = 'categories_places365.txt'
        if not os.access(file_name_category, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
            os.system('wget ' + synset_url)
        classes = list()
        with open(file_name_category) as class_file:
            for line in class_file:
                classes.append(line.strip().split(' ')[0][3:])
        classes = tuple(classes)

        # indoor and outdoor relevant
        file_name_IO = 'IO_places365.txt'
        if not os.access(file_name_IO, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
            os.system('wget ' + synset_url)
        with open(file_name_IO) as f:
            lines = f.readlines()
            labels_IO = []
            for line in lines:
                items = line.rstrip().split()
                labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
        labels_IO = np.array(labels_IO)

        # scene attribute relevant
        file_name_attribute = 'labels_sunattribute.txt'
        if not os.access(file_name_attribute, os.W_OK):
            synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
            os.system('wget ' + synset_url)
        with open(file_name_attribute) as f:
            lines = f.readlines()
            labels_attribute = [item.rstrip() for item in lines]
        file_name_W = 'W_sceneattribute_wideresnet18.npy'
        if not os.access(file_name_W, os.W_OK):
            synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
            os.system('wget ' + synset_url)
        W_attribute = np.load(file_name_W)

        return classes, labels_IO, labels_attribute, W_attribute

    def hook_feature(self, module, input, output):
        #self.features_blobs.append(np.squeeze(output.data.cpu().numpy()))
        self.features_blobs.append(np.squeeze(output.data))
        return

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        # print(type(feature_conv), type(weight_softmax), type(class_idx))
        size_upsample = (256, 256)
        nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            # print(type(weight_softmax[class_idx]))
            cam = weight_softmax[class_idx].matmul(feature_conv.reshape((nc, h*w)))
            cam = cam.view(h, w)
            cam = cam - torch.min(cam)
            cam_img = cam / torch.max(cam)
            cam_img = np.uint8(255 * cam_img.cpu().numpy())
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def returnTF(self):
    # load the image transformer
        tf = trn.Compose([
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return tf

    def load_model(self):
        # this model has a last conv feature map as 14x14

        model_file = 'wideresnet18_places365.pth.tar'
        if not os.access(model_file, os.W_OK):
            os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
            os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

        import wideresnet
        model = wideresnet.resnet18(num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        model.eval()



        # the following is deprecated, everything is migrated to python36

        ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
        #from functools import partial
        #import pickle
        #pickle.load = partial(pickle.load, encoding="latin1")
        #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

        model.eval()
        # hook the feature extractor
        features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self.hook_feature)
        return model
    
    def load_image(self, img_name):
        img_unr = Image.open(img_name)
        img = V(self.returnTF()(img_unr).unsqueeze(0))
        self.img = img.to(self.device)
        self.img_name = img_name
        self.og_image = plt.imread(img_name)
        del self.features_blobs
        self.features_blobs = []
        return img, img_unr
    
    def set_image(self, image, img_name = 'None'):
        image = image.convert('RGB') 
        img = V(self.returnTF()(image).unsqueeze(0))
        self.img = img.to(self.device)
        self.img_name = img_name
        self.og_image = image.convert('RGB') 
        self.og_image = np.array(self.og_image) 
        del self.features_blobs
        self.features_blobs = []
        return img, image

    
    def run(self, supress_printing = True, supress_images = True):
        # get the softmax weight
    
        # forward pass
        logit = self.model.forward(self.img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        # get top 5 predictions
        top_five_pred = dict()
        for i in range(0, 5):
            top_five_pred[self.classes[idx[i]]] = probs[i]

        # get attributes of the image
        # print(type(self.W_attribute), type(self.features_blobs[1]))
        responses_attribute = torch.matmul(self.W_attribute, self.features_blobs[1].double())
        idx_a = torch.argsort(responses_attribute)
        attributes = [self.labels_attribute[idx_a[i].item()] for i in range(-1,-10,-1)]
        CAMs = self.returnCAM(self.features_blobs[0], self.weight_softmax, [idx[0]])

        if not supress_printing:
            print('RESULT ON ' + self.img_name)
            # output the IO prediction
            io_image = np.mean(self.labels_IO[idx[:10]]) # vote for the indoor or outdoor
            if io_image < 0.5:
                print('--TYPE OF ENVIRONMENT: indoor')
            else:
                print('--TYPE OF ENVIRONMENT: outdoor')
            
            # output the prediction of scene category
            print('--SCENE CATEGORIES:')
            
            for i in range(0, 5):
                print('{:.3f} -> {}'.format(probs[i], self.classes[idx[i]]))
                
            # output the scene attributes
            print('--SCENE ATTRIBUTES:')
            print(', '.join(attributes))

            # generate class activation mapping
            print('Class activation map is saved as cam.jpg')
        
        

        # render the CAM and output
        if not supress_images:
            #self.og_image = cv2.cvtColor(self.og_image, cv2.COLOR_BGR2RGB)
            height, width, _ = self.og_image.shape
            heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)

            result = (heatmap * 0.4 + self.og_image * 0.5)/255
            
            #show images
            fig, axs = plt.subplots(2, 1, figsize=(13, 7.5))
            axs[0].imshow(self.og_image)
            axs[0].axis('off')
            axs[1].imshow(result)
            axs[1].axis('off')
            plt.show()
        return top_five_pred, attributes

		
		
if __name__ == '__main__':
    # load the test image
    img_url = 'http://places.csail.mit.edu/demo/6.jpg'
    os.system('wget %s -q -O test.jpg' % img_url)

    img_nam = 'test.jpg'
    x = SceneDetectionClass()
    x.load_image(img_nam)
    top_pred, attributes = x.run(supress_printing = False, supress_images = False)

    print(top_pred, attributes)
