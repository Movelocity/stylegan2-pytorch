import numpy as np
import torch


latents = torch.from_numpy(np.random.randn(1, 512).astype('float32'))
truncation = torch.tensor(0.5)

model = torch.load('twdne3_g.pt')

with open('twdne3.onnx', 'wb') as out_f:
    torch.onnx.export(
        model=model, 
        args=([latents], truncation),
        export_params=True,
        f=out_f,
        verbose=True,
        training=False,
        input_names=['latents', 'truncation'],
        output_names=['images'],
        opset_version=10,
    )

