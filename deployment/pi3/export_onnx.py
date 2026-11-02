"""Export trusted YOLO11n checkpoint to static FP32 ONNX and check parity."""
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    import onnx
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO
    from prepare_model import NAMES
    root = Path(__file__).resolve().parent
    checkpoint = root / 'models/best.pt'
    model = YOLO(str(checkpoint))
    if [model.names[i] for i in range(len(model.names))] != NAMES:
        raise ValueError('Checkpoint taxonomy does not match deployment')
    target = Path(model.export(format='onnx', imgsz=640, batch=1, dynamic=False,
                             half=False, simplify=False, opset=17, device='cpu', nms=False))
    graph = onnx.load(str(target))
    onnx.checker.check_model(graph)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(target), sess_options=options, providers=['CPUExecutionProvider'])
    data = np.random.default_rng(42).random((1, 3, 640, 640), dtype=np.float32)
    with torch.inference_mode():
        result = model.model(torch.from_numpy(data))
        expected = (result[0] if isinstance(result, (tuple, list)) else result).detach().numpy()
    actual = session.run(None, {session.get_inputs()[0].name: data})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    report = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'onnx_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'input': list(data.shape), 'output': list(actual.shape), 'names': NAMES,
              'opset': 17, 'precision': 'FP32', 'nms': False,
              'parity_max_abs_error': float(np.max(np.abs(actual-expected))),
              'validation': 'ONNX checker and CPU PT/ORT parity on seeded synthetic input; not accuracy evaluation'}
    target.with_suffix('.onnx.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
