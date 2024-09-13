import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import random
import torchvision.transforms.functional as F
import torchvision.transforms.functional as TF

class JointRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, heatmap):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(heatmap)
        return image, heatmap

class JointRandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, heatmap):
        angle = random.uniform(-self.degrees, self.degrees)
        return F.rotate(image, angle), F.rotate(heatmap, angle)

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, heatmap):
        for t in self.transforms:
            image, heatmap = t(image, heatmap)
        return image, heatmap

class GolfBallDataset(Dataset):
    def __init__(self, root_dir, purpose, split='train', image_size=256, group_by_video=False, transform=None, joint_transform=None):
        self.root_dir = root_dir
        self.purpose = purpose
        self.image_size = image_size
        self.group_by_video = group_by_video
        self.transform = transform
        self.joint_transform = joint_transform
        
        if purpose == 'detection':
            self.split = split
        elif purpose == 'tracking':
            if split != 'test':
                print("Warning: Tracking data is only for testing. Ignoring split parameter.")
            self.split = 'test'
        else:
            raise ValueError("Purpose must be either 'detection' or 'tracking'")
        
        self.image_dir = os.path.join(root_dir, purpose, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, purpose, 'Annotations')
        self.image_list = self._load_image_list()
        
        if self.group_by_video and self.purpose == 'tracking':
            self.video_frames = self._group_frames_by_video()

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.resize_transform = transforms.Resize((image_size, image_size))

    def _load_image_list(self):
        if self.purpose == 'detection':
            if self.split == 'train':
                image_set_file = os.path.join(self.root_dir, self.purpose, 'ImageSets', 'Main', 'train.txt')
            elif self.split == 'val':
                image_set_file = os.path.join(self.root_dir, self.purpose, 'ImageSets', 'Main', 'test.txt')
            else:  # 'test'
                image_set_file = os.path.join(self.root_dir, self.purpose, 'ImageSets', 'Main', 'test_video.txt')
            
            with open(image_set_file, 'r') as f:
                return [line.strip() for line in f]
        
        else:  # 'tracking'
            image_set_dir = os.path.join(self.root_dir, self.purpose, 'ImageSets', 'Main')
            all_images = []
            for txt_file in os.listdir(image_set_dir):
                if txt_file.endswith('.txt'):
                    video_name = txt_file[:-4]  # Remove '.txt'
                    with open(os.path.join(image_set_dir, txt_file), 'r') as f:
                        frames = [line.strip() for line in f]
                        # Prepend video name to each frame number
                        all_images.extend([f"{video_name}_{frame}" for frame in frames])
            print(f"Loaded {len(all_images)} images for tracking")
            print(f"First few image names: {all_images[:5]}")
            return all_images


    def _group_frames_by_video(self):
        video_frames = {}
        for img_name in self.image_list:
            video_name = img_name.split('_')[0]
            if video_name not in video_frames:
                video_frames[video_name] = []
            video_frames[video_name].append(img_name)
        return video_frames

    def __len__(self):
        if self.group_by_video and self.purpose == 'tracking':
            return len(self.video_frames)
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        
        if self.purpose == 'tracking':
            parts = img_name.split('_')
            video_name = parts[0] + '_' + parts[1]
            frame_number = parts[-1]
            
            if video_name.startswith('Put'):
                folder_name = video_name
                file_name = f"{video_name[:-2]}{parts[1]}_{frame_number}.jpg"
                ann_file_name = f"{video_name[:-2]}{parts[1]}_{frame_number}.xml"
            elif video_name.startswith('Golf'):
                folder_name = video_name
                file_name = f"{parts[1].zfill(2)}_{frame_number}.jpg"
                ann_file_name = f"{parts[1].zfill(2)}_{frame_number}.xml"
            else:
                raise ValueError(f"Unknown video name format: {video_name}")
            
            img_path = os.path.join(self.image_dir, folder_name, file_name)
            ann_path = os.path.join(self.annotation_dir, ann_file_name)
        else:
            file_name = f"{img_name}.jpg"
            img_path = os.path.join(self.image_dir, file_name)
            ann_path = os.path.join(self.annotation_dir, f"{img_name}.xml")

        image = Image.open(img_path).convert('RGB')
        bbox = self._parse_annotation(ann_path)

        heatmap = self._generate_heatmap(image.size, bbox)
        heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))  # Convert to PIL Image

        if self.joint_transform:
            image, heatmap = self.joint_transform(image, heatmap)

        # Apply the same random resized crop to both image and heatmap
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            image, scale=(0.8, 1.0), ratio=(1.0, 1.0))
        
        # Check if the crop contains a significant part of the golf ball
        heatmap_np = np.array(heatmap)
        crop_heatmap = heatmap_np[j:j+h, i:i+w]
        if crop_heatmap.max() > 0.5 * heatmap_np.max():  # Ensure the crop contains at least 50% of the max heatmap value
            image = TF.resized_crop(image, i, j, h, w, (self.image_size, self.image_size))
            heatmap = TF.resized_crop(heatmap, i, j, h, w, (self.image_size, self.image_size))
        else:
            # If the crop doesn't contain enough of the golf ball, just resize
            image = TF.resize(image, (self.image_size, self.image_size))
            heatmap = TF.resize(heatmap, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
        
        heatmap = TF.to_tensor(heatmap)

        return image, heatmap, file_name
    
    def _process_single_frame(self, img_name):
        if self.purpose == 'tracking':
            parts = img_name.split('_')
            video_name = parts[0] + '_' + parts[1]  # 'Put_1' or 'Golf_2'
            frame_number = parts[-1]  # '001'
            
            if video_name.startswith('Put'):
                folder_name = video_name  # 'Put_1'
                file_name = f"{video_name[:-2]}{parts[1]}_{frame_number}.jpg"  # 'Put1_001.jpg'
                ann_file_name = f"{video_name[:-2]}{parts[1]}_{frame_number}.xml"  # 'Put1_001.xml'
            elif video_name.startswith('Golf'):
                folder_name = video_name  # 'Golf_2'
                file_name = f"{parts[1].zfill(2)}_{frame_number}.jpg"  # '02_001.jpg'
                ann_file_name = f"{parts[1].zfill(2)}_{frame_number}.xml"  # '02_001.xml'
            else:
                raise ValueError(f"Unknown video name format: {video_name}")
            
            img_path = os.path.join(self.image_dir, folder_name, file_name)
            ann_path = os.path.join(self.annotation_dir, ann_file_name)
        else:
            img_path = os.path.join(self.image_dir, f"{img_name}.jpg")
            ann_path = os.path.join(self.annotation_dir, f"{img_name}.xml")


        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            print(f"Directory contents: {os.listdir(os.path.dirname(img_path))}")
            raise FileNotFoundError(f"No such file: {img_path}")

        if not os.path.exists(ann_path):
            print(f"Annotation file not found: {ann_path}")
            print(f"Directory contents: {os.listdir(os.path.dirname(ann_path))}")
            raise FileNotFoundError(f"No such file: {ann_path}")

        image = Image.open(img_path).convert('RGB')
        bbox = self._parse_annotation(ann_path)

        heatmap = self._generate_heatmap(image.size, bbox)

        image = self.transform(image)
        heatmap = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor()
        ])(heatmap)

        return image, heatmap

    def _process_video(self, frames):
        video_data = []
        for frame in frames:
            video_data.append(self._process_single_frame(frame))
        return video_data

    def _parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        
        obj = root.find('object')
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        return [xmin, ymin, xmax, ymax]

    def _generate_heatmap(self, image_size, bbox, sigma=8):
        heatmap = np.zeros(image_size[::-1], dtype=np.float32)
        
        xmin, ymin, xmax, ymax = bbox
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        
        x = np.arange(0, image_size[0], 1, np.float32)
        y = np.arange(0, image_size[1], 1, np.float32)[:, np.newaxis]
        
        heatmap = np.exp(-((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * sigma ** 2))
        
        return heatmap