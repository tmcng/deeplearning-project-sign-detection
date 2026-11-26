# faster_rcnn/dataset.py
"""COCO-format dataset loader cho mô hình Faster R-CNN nhận diện biển báo giao thông."""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import torch
from torch.utils.data import Dataset

# Tắt đa luồng nội bộ OpenCV để tránh xung đột tài nguyên CPU khi DataLoader dùng num_workers > 0
cv2.setNumThreads(0)

# Các hằng số cấu hình data augmentation an toàn (không đổi ngữ nghĩa biển báo)
AUG_ALPHA_MIN: float = -0.15
AUG_ALPHA_MAX: float = 0.15
AUG_BETA_MIN: float = -15.0
AUG_BETA_MAX: float = 15.0


class ZaloCOCODataset(Dataset):
    """Dataset COCO cho bài toán nhận diện biển báo giao thông (Faster R-CNN).

    Hành vi:
        - is_train=True: Trả về (image_tensor, target) với boxes, labels và image_id.
        - is_train=False: Trả về (image_tensor, image_id) phục vụ inference/test.
        - use_augment=True: Chỉ áp dụng biến đổi độ sáng và tương phản ngẫu nhiên (Safe Augmentation).
          LƯU Ý: Không dùng Horizontal Flip vì làm sai lệch biển báo cấm rẽ, chỉ hướng, tốc độ.
    """

    def __init__(
        self,
        json_path: Union[str, os.PathLike],
        img_dir: Union[str, os.PathLike],
        is_train: bool = True,
        use_augment: bool = False,
    ) -> None:
        """Khởi tạo dataset từ file COCO JSON và thư mục ảnh.

        Args:
            json_path: Đường dẫn đến file annotation JSON định dạng COCO.
            img_dir: Thư mục chứa các tệp ảnh tương ứng.
            is_train: True nếu là chế độ huấn luyện/tính loss, False nếu chỉ trích xuất ảnh inference.
            use_augment: True để kích hoạt safe brightness/contrast augmentation.
        """
        self.img_dir: str = str(img_dir)
        self.is_train: bool = is_train
        self.use_augment: bool = use_augment

        with open(json_path, "r", encoding="utf-8") as f:
            self.coco_data: Dict[str, Any] = json.load(f)

        self.images: List[Dict[str, Any]] = self.coco_data.get("images", [])

        if self.is_train:
            self.img_to_anns: Dict[int, List[Dict[str, Any]]] = {}
            for ann in self.coco_data.get("annotations", []):
                self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        """Tổng số ảnh trong dataset."""
        return len(self.images)

    def _apply_safe_augment(self, image: cv2.typing.MatLike) -> cv2.typing.MatLike:
        """Áp dụng safe augmentation: thay đổi độ sáng và độ tương phản nhẹ."""
        alpha = 1.0 + random.uniform(AUG_ALPHA_MIN, AUG_ALPHA_MAX)
        beta = random.uniform(AUG_BETA_MIN, AUG_BETA_MAX)
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[Dict[str, torch.Tensor], int]]:
        """Lấy một mẫu dữ liệu theo chỉ mục.

        Args:
            idx: Chỉ mục của ảnh trong dataset.

        Returns:
            Tuple (image_tensor, target) nếu is_train=True, hoặc (image_tensor, image_id) nếu is_train=False.
        """
        img_info = self.images[idx]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Không đọc được ảnh '{img_path}'. Kiểm tra IMG_DIR và file_name trong COCO JSON."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Chế độ inference: chỉ trả về tensor ảnh và image_id
        if not self.is_train:
            image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            return image_tensor, img_info["id"]

        anns = self.img_to_anns.get(img_info["id"], [])
        boxes: List[List[float]] = []
        labels: List[int] = []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann["category_id"])

        if self.use_augment and len(boxes) > 0:
            image = self._apply_safe_augment(image)

        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        target: Dict[str, torch.Tensor] = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([img_info["id"]]),
        }

        image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        return image_tensor, target


def collate_fn(batch: List[Any]) -> Tuple[Any, ...]:
    """Hàm gộp batch tùy chỉnh cho DataLoader để xử lý danh sách tensor kích thước biến thiên."""
    return tuple(zip(*batch))
