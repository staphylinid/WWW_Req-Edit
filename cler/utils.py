"""Atomic files, checkpoint state, logging and reproducible random states."""
from pathlib import Path
import hashlib
import json
import logging
import random
import sys
import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path)


def trainable_state(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model, values):
    expected = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if set(expected) != set(values):
        raise ValueError('Checkpoint parameter names differ from this editor')
    with torch.no_grad():
        for name, parameter in expected.items():
            if parameter.shape != values[name].shape:
                raise ValueError('Checkpoint shape differs: ' + name)
            parameter.copy_(values[name].to(parameter))


def frozen_parameters(model):
    return (model.backbone.original_parameters if hasattr(model.backbone, 'original_parameters')
            else model.original_parameters)


def frozen_digest(model):
    result = hashlib.sha256()
    for parameter in frozen_parameters(model):
        if parameter.requires_grad:
            raise ValueError('Original parameter is trainable')
        result.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def logger(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    value = logging.getLogger(str(folder.resolve()))
    value.setLevel(logging.INFO)
    value.propagate = False
    if not value.handlers:
        for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(folder/'run.log',encoding='utf-8')):
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            value.addHandler(handler)
    return value


def bind(folder, protocol):
    path = Path(folder)/'protocol.json'
    protocol = json.loads(json.dumps(protocol, ensure_ascii=False))
    if path.exists() and read(path) != protocol:
        raise ValueError('Existing run has a different protocol: ' + str(path))
    write(path, protocol)
