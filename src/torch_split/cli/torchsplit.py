import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Torch Split CLI")
    parser.add_argument("--version", action="version", version="torch-split 0.1.0")
    args = parser.parse_args()
