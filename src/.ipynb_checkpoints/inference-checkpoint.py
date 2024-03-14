import json
import os
import io
import logging

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model = ASRModel.restore_from(os.path.join(model_dir, 'reazonspeech-nemo-v2.nemo'))
    model = model.to(DEVICE)
    return {'model': model}


def input_fn(request_body, request_content_type):
    """
    Takes in request and transforms it to necessary input type
    """
    int16pcm = np.load(io.BytesIO(request_body))
    float32pcm = int16pcm.astype(np.float32)
    float32pcm /= 32767

    return float32pcm


def predict_fn(input_data, model_dict):
    """
    SageMaker model server invokes `predict_fn` on the return value of `input_fn`.

    Return predictions
    """
    result = model_dict['model'].transcribe(input_data)
    return result[0][0]


def output_fn(predictions, content_type):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    return predictions
