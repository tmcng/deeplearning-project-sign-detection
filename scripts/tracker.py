"""
scripts/tracker.py
Mô-đun quản lý theo dõi thí nghiệm bằng Weights & Biases (W&B).
Đăng nhập trực tiếp bằng API Key, chạy Local bình thường.
"""

from pathlib import Path


class WandbTracker:
    def __init__(self, api_key=None, project_name="Zalo-Traffic-Sign-FasterRCNN", run_name=None, config_dict=None, enabled=True):
        self.enabled = enabled
        self.run = None

        if not self.enabled:
            print("[TRACKER] Che do W&B dang tat (disabled).")
            return

        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("[CANH BAO] Thu vien wandb chua duoc cai dat. Bo qua tracking.")
            self.enabled = False
            return

        # Đăng nhập trực tiếp bằng API Key nếu có truyền vào hoặc đọc từ môi trường
        import os
        key = api_key or os.environ.get("WANDB_API_KEY")
        if key:
            self.wandb.login(key=key)

        # Khởi tạo Run
        try:
            self.run = self.wandb.init(
                project=project_name,
                name=run_name,
                config=config_dict or {},
                reinit=True
            )
            print(f"[TRACKER] W&B Run da khoi tao: {self.run.name} (Project: {project_name})")
        except Exception as e:
            print(f"[CANH BAO] Khong the khoi tao W&B ({e}). Chuyen sang che do khong log.")
            self.enabled = False

    def log_metrics(self, metrics_dict, step=None):
        """Ghi nhận loss, learning rate, mAP theo step hoặc epoch."""
        if self.enabled and self.run is not None:
            self.wandb.log(metrics_dict, step=step)

    def log_artifact(self, file_path, artifact_name, artifact_type="model", description=""):
        """Lưu trữ file trọng số .pth hoặc file báo cáo .json lên W&B."""
        if not self.enabled or self.run is None:
            return

        p = Path(file_path)
        if not p.exists():
            print(f"[CANH BAO] File khong ton tai de upload artifact: {file_path}")
            return

        try:
            artifact = self.wandb.Artifact(name=artifact_name, type=artifact_type, description=description)
            if p.is_file():
                artifact.add_file(str(p))
            elif p.is_dir():
                artifact.add_dir(str(p))
            self.run.log_artifact(artifact)
            print(f"[TRACKER] Da upload artifact '{artifact_name}' ({p.name}) len W&B.")
        except Exception as e:
            print(f"[CANH BAO] Loi khi upload artifact: {e}")

    def finish(self):
        """Kết thúc phiên log W&B an toàn."""
        if self.enabled and self.run is not None:
            self.wandb.finish()
            print("[TRACKER] Da hoan tat va dong session W&B.")
