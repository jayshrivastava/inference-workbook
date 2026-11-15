"""Bridge the workbook checker to torch-pyodide's current API subset."""

import numpy as np
import torch

import browser_runner


def _patch_torch() -> None:
    if getattr(torch.Tensor, "_workbook_patched", False):
        return
    torch.Size = tuple

    class Testing:
        @staticmethod
        def assert_close(actual, expected, *, rtol=1e-5, atol=1e-8):
            if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
                raise AssertionError(f"Tensors are not close: {actual} != {expected}")

    torch.testing = Testing()
    original_getitem = torch.Tensor.__getitem__

    def getitem(self, key):
        if not isinstance(key, tuple) or all(part is not None for part in key):
            return original_getitem(self, key)
        array = np.asarray(self.tolist())
        return torch.tensor(array[key].tolist(), dtype=self.dtype, device=self.device)

    def setitem(self, key, value):
        array = np.asarray(self.tolist()).copy()
        if isinstance(value, torch.Tensor):
            value = value.tolist()
        if isinstance(key, tuple):
            key = tuple(np.asarray(part.tolist()) if isinstance(part, torch.Tensor) else part for part in key)
        elif isinstance(key, torch.Tensor):
            key = np.asarray(key.tolist())
        array[key] = value
        self._set(torch.tensor(array.tolist(), dtype=self.dtype, device=self.device))

    torch.Tensor.__getitem__ = getitem
    torch.Tensor.__setitem__ = setitem
    torch.Tensor._workbook_patched = True


def run(payload_json: str) -> str:
    _patch_torch()
    return browser_runner.run(payload_json)
