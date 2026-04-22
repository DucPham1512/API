from pathlib import Path
from typing import Union

import numpy as np
import onnxruntime as ort


class OnnxInferencer:
    """Wraps an ONNX Runtime session for VSR inference.

    Args:
        model_path: Path to the .onnx model file.
    """

    def __init__(self, model_path: Union[str, Path]):
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def predict(self, tensor: np.ndarray) -> np.ndarray:
        """Run inference on a preprocessed window tensor.

        Args:
            tensor: float32 array of shape (T, H, W, C) produced by the preprocessor.

        Returns:
            Model output array.
        """
        batch = tensor[np.newaxis]  # (1, T, H, W, C)
        outputs = self._session.run([self._output_name], {self._input_name: batch})
        return outputs[0]
