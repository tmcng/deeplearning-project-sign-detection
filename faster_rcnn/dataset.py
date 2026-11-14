# faster_rcnn/dataset.py

import os
import json
import torch
import cv2
import random
from torch.utils.data import Dataset

# Tắt đa luồng nội bộ OpenCV để tránh xung đột tài nguyên CPU khi DataLoader dùng num_workers > 0
cv2.setNumThreads(0)


class ZaloCOCODataset(Dataset):
    """
    Dataset cho Faster R-CNN.
    - is_train=True : trả về (image_tensor, target).
    - is_train=False: trả về (image_tensor, image_id) cho inference.
    - use_augment=True: Random Flip + Brightness/Contrast.
    """
    def __init__(self, json_path, img_dir, is_train=True, use_augment=False):
        self.img_dir     = img_dir
        self.is_train    = is_train
        self.use_augment = use_augment

        with open(json_path, 'r', encoding='utf-8') as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data['images']

        if self.is_train:
            self.img_to_anns = {}
            for ann in self.coco_data.get('annotations', []):
                self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Khong doc duoc anh '{img_path}'. Kiem tra IMG_DIR va file_name trong COCO JSON."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.is_train:
            image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return image_tensor, img_info['id']

        anns   = self.img_to_anns.get(img_info['id'], [])
        boxes  = []
        labels = []

        h_img, w_img = image.shape[:2]
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'])

        if self.use_augment and len(boxes) > 0:
            # Lưu ý an toàn cho biển báo giao thông:
            # KHÔNG lật ngang (flip) vì làm đổi chiều biển cấm rẽ, biển chỉ hướng và lộn ngược số biển tốc độ.
            # Chỉ áp dụng biến đổi độ sáng và độ tương phản nhẹ:
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta  = random.uniform(-15, 15)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        if len(boxes) == 0:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor  = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "image_id": torch.tensor([img_info['id']]),
        }

        image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


