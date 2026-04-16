import torch


class AmpKit:
    """
    A utility class to handle Automatic Mixed Precision (AMP) in PyTorch.
    It provides backward compatibility to support both newer and older PyTorch versions.
    """

    def __init__(self, device_type: str = "cuda"):
        """
        Initialize the AMP toolkit.
        Sets up the gradient scaler and autocast context manager based on the provided device type.
        """
        self.enabled = device_type == "cuda"
        try:
            # Newer PyTorch API
            from torch import amp as _amp

            self.grad_scaler = _amp.GradScaler(device_type, enabled=self.enabled)
            self._autocast_ctx = lambda: _amp.autocast(
                device_type=device_type, dtype=torch.float16
            )
            self._new_api = True
        except Exception:
            # Older PyTorch API fallback
            from torch.cuda.amp import GradScaler as _GradScaler, autocast as _autocast

            self.grad_scaler = _GradScaler(enabled=self.enabled)
            self._autocast_ctx = _autocast
            self._new_api = False

    def autocast(self):
        """
        Return the appropriate autocast context manager for mixed precision operations.
        Usage: `with amp_kit.autocast():`
        """
        return self._autocast_ctx()

    @property
    def scaler(self):
        """
        Property that returns the underlying GradScaler instance used for scaling gradients.
        """
        return self.grad_scaler
