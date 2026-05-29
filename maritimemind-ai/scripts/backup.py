import os
import shutil
from datetime import datetime

def backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backups/backup_{timestamp}"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    if os.path.exists("vector_store"):
        shutil.copytree("vector_store", os.path.join(backup_dir, "vector_store"))
    
    if os.path.exists("data/metadata"):
        shutil.copytree("data/metadata", os.path.join(backup_dir, "metadata"))
        
    print(f"Backup completed successfully to {backup_dir}")

if __name__ == "__main__":
    backup()
