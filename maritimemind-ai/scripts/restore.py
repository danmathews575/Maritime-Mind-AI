import os
import shutil
import sys

def restore(backup_dir):
    if not os.path.exists(backup_dir):
        print(f"Error: Backup directory '{backup_dir}' does not exist.")
        sys.exit(1)
        
    vs_backup = os.path.join(backup_dir, "vector_store")
    metadata_backup = os.path.join(backup_dir, "metadata")
    
    if os.path.exists(vs_backup):
        if os.path.exists("vector_store"):
            shutil.rmtree("vector_store")
        shutil.copytree(vs_backup, "vector_store")
        print("Restored vector_store.")
        
    if os.path.exists(metadata_backup):
        if os.path.exists("data/metadata"):
            shutil.rmtree("data/metadata")
        shutil.copytree(metadata_backup, "data/metadata")
        print("Restored metadata.")
        
    print("Restore completed successfully.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python restore.py <backup_directory_path>")
        sys.exit(1)
    restore(sys.argv[1])
