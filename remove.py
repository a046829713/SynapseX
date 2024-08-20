import os
import shutil

def delete_pycache(dir_path):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in dirs:
            if name == '__pycache__':
                full_path = os.path.join(root, name)                
                print(f"Deleting {full_path}...")
                shutil.rmtree(full_path)

if __name__ == "__main__":
    # 假設你想從當前工作目錄開始刪除所有__pycache__文件夾
    delete_pycache('.')