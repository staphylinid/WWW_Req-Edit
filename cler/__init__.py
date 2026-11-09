"""Req-Edit: counterfactual localization, local editing and history protection."""
from .backbone import SASRecBackbone, SequentialBackbone, create_sasrec, load_sasrec
from .model import create_editor
from .localization import locate, project_requests
from .projection import SemanticProjection, fit_projection, save_projection, load_projection
from .loss import click_loss, editor_loss
from .protection import protected_history, soft_project

__all__ = [
    'SASRecBackbone', 'SequentialBackbone', 'create_sasrec', 'load_sasrec',
    'create_editor', 'locate', 'project_requests', 'click_loss', 'editor_loss',
    'protected_history', 'soft_project',
    'SemanticProjection', 'fit_projection', 'save_projection', 'load_projection',
]
