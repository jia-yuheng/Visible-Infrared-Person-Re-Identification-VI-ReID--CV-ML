import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir, model=None, transform1=None, transform2=None, colorIndex = None, thermalIndex = None):
        
        data_dir = '/root/autodl-tmp/SYSU-MM01'
        # Load training images (path) and labels
        train_color_image_gobal = np.load(data_dir + 'train_rgb_resized_img.npy')
        train_color_image_part = np.load(data_dir + 'train_rgb_resized_img_part.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image_gobal = np.load(data_dir + 'train_ir_resized_img.npy')
        train_thermal_image_part = np.load(data_dir + 'train_ir_resized_img_part.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # BGR to RGB
        self.train_color_image_gobal = train_color_image_gobal
        self.train_color_image_part = train_color_image_part
        self.train_thermal_image_gobal = train_thermal_image_gobal
        self.train_thermal_image_part = train_thermal_image_part
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.model = model
       # print(self.model)
       # print(self.transform2)

    def __getitem__(self, index):

        img1_gobal, img1_part, target1 = self.train_color_image_gobal[self.cIndex[index]], self.train_color_image_part[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2_gobal, img2_part, target2 = self.train_thermal_image_gobal[self.tIndex[index]], self.train_thermal_image_part[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]


        img1 = self.transform1(img1_gobal)
        img2 = self.transform1(img2_gobal)
        upper_part1 = self.transform2(img1_part)
        upper_part2 = self.transform2(img2_part)

       # print(img2.shape)
       # img2 = self.model(img2)

        return img1, img2, upper_part1, upper_part2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform1=None, transform2=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = '/root/autodl-tmp/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image_gobal = []
        train_color_image_part = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            #img = img.convert('L')
            img = img.convert('RGB')
            img_gobal = img.resize((144, 388), Image.ANTIALIAS)
            pix_array = np.array(img_gobal)
            train_color_image_gobal.append(pix_array)

            width = img.size[0]

            height = img.size[1] / 2
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, 194), Image.ANTIALIAS)
            
             # 计算图像总高度的30%
            '''
            height = img.size[1] * 0.3 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.3 / 0.5)), Image.ANTIALIAS)  # 调整高度比例
            '''
            '''
            # 计算图像总高度的60%
            height = img.size[1] * 0.6 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.6 / 0.5)), Image.ANTIALIAS)  # 调整高度比例
            '''
            
            pix_array_part = np.array(img_part)
            train_color_image_part.append(pix_array_part)
        train_color_image_gobal = np.array(train_color_image_gobal)
        train_color_image_part = np.array(train_color_image_part)
        
        train_thermal_image_gobal = []
        train_thermal_image_part = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            #img = img.convert('L')
            img = img.convert('RGB')
            img_gobal = img.resize((144, 388), Image.ANTIALIAS)
            pix_array = np.array(img_gobal)
            train_thermal_image_gobal.append(pix_array)

            width = img.size[0]
             # 计算图像总高度的50%

            height = img.size[1] / 2
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, 194), Image.ANTIALIAS)

            # 计算图像总高度的30%
            '''
            height = img.size[1] * 0.3 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.3 / 0.5)), Image.ANTIALIAS)  # 调整高度比例'''
            '''
            # 计算图像总高度的60%
            height = img.size[1] * 0.6 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.6 / 0.5)), Image.ANTIALIAS)  # 调整高度比例'''

            pix_array_part = np.array(img_part)
            train_thermal_image_part.append(pix_array_part)
        train_thermal_image_part = np.array(train_thermal_image_part)
        
        # BGR to RGB
        self.train_color_image_gobal = train_color_image_gobal
        self.train_color_image_part = train_color_image_part
        self.train_color_label = train_color_label
        
        # BGR to RGB
        self.train_thermal_image_gobal = train_thermal_image_gobal
        self.train_thermal_image_part = train_thermal_image_part
        self.train_thermal_label = train_thermal_label
        
        self.transform1 = transform1
        self.transform2 = transform2
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1_gobal, img1_part, target1 = self.train_color_image_gobal[self.cIndex[index]], self.train_color_image_part[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2_gobal, img2_part, target2 = self.train_thermal_image_gobal[self.tIndex[index]], self.train_thermal_image_part[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1_gobal = self.transform1(img1_gobal)
        img2_gobal = self.transform1(img2_gobal)
        img1_part = self.transform2(img1_part)
        img2_part = self.transform2(img2_part)


        return img1_gobal, img2_gobal, img1_part, img2_part, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform1=None, transform2=None, img_size = (144,388)):

        test_image_gobal = []
        test_image_part = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            #img = img.convert('RGB')
            #img = img.convert('L')
            img = img.convert('RGB')
            img_gobal = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img_gobal)
            test_image_gobal.append(pix_array)

            width = img.size[0]
            # 计算图像总高度的50%

            height = img.size[1] / 2
            img_part = img.crop((0, 0, width, height))
            img_part = img_part.resize((144, 194), Image.ANTIALIAS)

            # 计算图像总高度的30%
            '''
            height = img.size[1] * 0.3 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.3 / 0.5)), Image.ANTIALIAS)  # 调整高度比例'''

            '''
            # 计算图像总高度的60%
            height = img.size[1] * 0.6 
            upper_part = img.crop((0, 0, width, height))
            img_part = upper_part.resize((144, int(194 * 0.6 / 0.5)), Image.ANTIALIAS)  # 调整高度比例  int(194 * 0.6 / 0.5) = 232
            '''
            pix_array = np.array(img_part)
            test_image_part.append(pix_array)

        test_image_gobal = np.array(test_image_gobal)
        test_image_part = np.array(test_image_part)


        self.test_image_gobal = test_image_gobal
        self.test_image_part = test_image_part
        self.test_label = test_label
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):
        img_gobal,  target1 = self.test_image_gobal[index],  self.test_label[index]
        img_part = self.test_image_part[index]

        img_gobal = self.transform1(img_gobal)
        img_part = self.transform2(img_part)
        return img_gobal, img_part, target1

    def __len__(self):
        return len(self.test_image_gobal)

def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


