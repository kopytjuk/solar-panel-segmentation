from pathlib import Path


def is_directory_not_empty(directory_path: Path) -> bool:
    path = Path(directory_path)
    # Check if the path exists and is a directory
    if path.exists() and path.is_dir():
        # Check if the directory is not empty
        if any(path.iterdir()):
            return True
    return False