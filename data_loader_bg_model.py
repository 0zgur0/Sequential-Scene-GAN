import torch
from torch.utils.data import  Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os
import torchvision
from utils import imshow    

class CocoData(Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        category_names : name of the categories desired dataset consists
        final_img_size : Dataset image size, default: 128
        
        
        Return: 
            'image'  : 3x128x128
            'segmentation mask' : num_catx128x128  --- only one  instance for specific category (one instance for each category)
            'category' : multiple categories (e.g. zebra, giraffe)
        
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, category_names = None, final_img_size=128):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.final_img_size = final_img_size     
        self.transform2 = transforms.Compose([
                                               transforms.Scale((final_img_size,final_img_size)),
                                               transforms.ToTensor(),
                                           ])
    
        
        if category_names == None:
            self.category = None
            self.ids = list(self.coco.imgs.keys())
        else:
            self.category = self.coco.getCatIds(catNms=category_names) #e.g. [22,25]
            
            self.ids = []
            self.cat = []
            for x in self.category:
                self.ids +=  self.coco.getImgIds(catIds=x )
                self.cat +=  [x]*len(self.coco.getImgIds(catIds=x )) #e.g. [22,22,...,22]

            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img_size = img.size
        img_size_x = img_size[0]
        img_size_y = img_size[1]
        seg_masks = torch.zeros([len(self.category),self.final_img_size,self.final_img_size])
        instance_types = []
        
        for i in range(len(target)):    
            instance = target[i]
            instance_types.append(instance['category_id'])

        idx_list = [i for i in range(len(instance_types)) if ((instance_types[i] in self.category) and (len(target[i]['segmentation']))>=1)]
        num_object = len(idx_list)

        for i in range(num_object): 
            idx = idx_list[np.random.choice(len(idx_list),1)[0]]
            idx_list.remove(idx)
            instance = target[idx]

            mask = Image.new('L', (img_size_x, img_size_y))
            for j in range(len(instance['segmentation'])):
                poly = instance['segmentation'][j]
                ImageDraw.Draw(mask).polygon(poly, outline=1, fill=1)
            
            mask= self.transform2(mask)
            if torch.max(mask) != 0:
                mask = mask/torch.max(mask)
            
            seg_masks[self.category.index(instance['category_id']),:,:] += mask
            
        if self.transform is not None:
            img = self.transform(img)



        seg_masks = torch.clamp(seg_masks,0,1)
        
        sample = {'image': img, 'seg_mask': seg_masks, 'cat': self.cat[index], 'num_object':num_object,'id':img_id}
        return sample

    def __len__(self):
        return len(self.ids)

    def discard_small(self, min_area, max_area=1):
        temp = []
        for img_id in self.ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            instance_types = []
            valid_mask = False
            
            path = self.coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.root, path))
            img_size = img.size
            img_size_x = img_size[0]
            img_size_y = img_size[1]

            total_fg_area = 0
            total_fg_area_relevant = 0
            
            for i in range(len(target)):    
                instance = target[i]
                total_fg_area += instance['area']
                instance_types.append(instance['category_id'])
                if instance['category_id'] in self.category and len(instance['segmentation'])>=1:
                    total_fg_area_relevant +=  instance['area']
                    valid_mask = True
                if (instance['category_id'] in self.category) and (type(instance['segmentation']) is not list):
                    valid_mask = False
                    break 

            if valid_mask and total_fg_area_relevant/(img_size_x*img_size_y) > min_area and total_fg_area/(img_size_x*img_size_y) < max_area:
                temp.append(img_id)
        
        print(str(len(self.ids)) + '-->' + str(len(temp)))
        self.ids = temp
        

    def discard_bad_examples(self, path):  
        file_list = open(path, "r")
        bad_examples = file_list.readlines()
        for i in range(len(bad_examples)):
            bad_examples[i] = int(bad_examples[i][:-1])

        temp = []
        for img_id in self.ids:
            if not (img_id in bad_examples):
                temp.append(img_id)
        
        print(str(len(self.ids)) + '-->' + str(len(temp)))
        self.ids = temp
        print('Bad examples are left out!')
        
#-------------------------Example-----------------------------------------
if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ])    
    dataset = CocoData(root = 'C:/Users/motur/coco/images/train2017',
                            annFile = 'C:/Users/motur/coco/annotations/instances_train2017.json',
                            category_names = ['elephant'],
                            transform=transform)
    
    dataset.discard_small(0.0, max_area= 1)
    dataset.discard_bad_examples('bad_examples_list.txt')
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False)   
    print('Number of samples: ', len(dataset))

    
    for num_iter, sample_batched in enumerate(train_loader,0):
        img_id = sample_batched['id'][0]
        image = dataset[num_iter]
        image = image['img']
        imshow(torchvision.utils.make_grid(image))
        plt.pause(0.001)
      
        