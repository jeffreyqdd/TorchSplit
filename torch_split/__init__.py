"""TorchSplit - Automatic model splitting for PyTorch."""


# # Try to import the Rust backend, handle case where it's not built yet (e.g. during linting)
# try:
#     from torch_split import _rust_backend
# except ImportError:
#     _rust_backend = None

# __all__ = ["SplitClient", "_rust_backend"]
