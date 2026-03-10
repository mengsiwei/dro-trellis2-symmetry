import objaverse
import shutil
from pathlib import Path

# 指定下载保存路径（可改为任意目录）
DOWNLOAD_DIR = Path(__file__).resolve().parent / "objects"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 选定你想要的 UID 列表
selected_uids = ["8476c4170df24cf5bbe6967222d1a42d", "8ff7f1f2465347cd8b80c9b206c2781e"]

# 下载模型（默认下载到库缓存，再复制到指定路径）
objects = objaverse.load_objects(
    uids=selected_uids,
    download_processes=4,
)

# 复制到指定目录，便于统一管理
for uid, src_path in objects.items():
    src = Path(src_path)
    dst = DOWNLOAD_DIR / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    else:
        dst = src
    print(f"UID: {uid} 已保存至: {dst}")