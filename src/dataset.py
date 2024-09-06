import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class GolfBallDataset(Dataset):
    def __init__(self, root_dir, purpose, split='train', image_size=256, group_by_video=False, transform=None):
        self.root_dir = root_dir
        self.purpose = purpose
        self.image_size = image_size
        self.group_by_video = group_by_video
        self.transform = transform
        
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
        if self.group_by_video and self.purpose == 'tracking':
            video_name = list(self.video_frames.keys())[idx]
            frames = self.video_frames[video_name]
            return self._process_video(frames)
        else:
            img_name = self.image_list[idx]
            return self._process_single_frame(img_name)

    def _process_single_frame(self, img_name):
        #print(f"Processing image: {img_name}")
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
        
        #print(f"Attempting to open image at: {img_path}")
        #print(f"Attempting to open annotation at: {ann_path}")
        
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