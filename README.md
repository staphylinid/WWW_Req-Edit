# Req-Edit

Learned-projection implementation of *Breaking Information Cocoons Proactively: A Request-Guided Recommendation Editing Framework with Causal Parameter Localization*.

## Installation

```sh
pip install -e .
```

Use `pip install -e ".[sasrec]"` for the SASRec adapter. GRU4Rec and Mamba4Rec use `SequentialBackbone`.

## Inputs

Supply a pretrained backbone, item and request text embeddings, train/validation/test rows, and validation/test candidate pools. Item IDs index both item tables; ID 0 is padding. Each row supplies `user_id`, `history`, `groups`, `query_row`, and `positive`. The `assets` object supplies `data.rows`, catalog size `data.n`, candidate `pools`, and the backbone checkpoint hash in `provenance["base_sha256"]`.

## SASRec workflow

```python
from pathlib import Path

from cler import SemanticProjection, create_editor, fit_projection, locate, save_projection
from cler.data import prepare_inputs
from cler.train import train

run_dir = Path("runs/req_edit")
backbone = editor_backbone  # SASRecBackbone with editable_scope="all_six"
projection = SemanticProjection(input_dim=1024, hidden_dim=128, output_dim=32).to(device)
fit_projection(projection, item_semantics, backbone.item_embeddings, assets.data.rows["train"], seed=seed)
save_projection(projection, run_dir / "projection.pt")

prepared = prepare_inputs(backbone, assets.data, queries, projection, 128, logger)
editor = create_editor(backbone, request_dim=1024, weights_per_block=512, selected_blocks=16, banks=4)
masks = locate(editor, prepared, config, seed, run_dir / "localization", logger)
result = train(editor, prepared, masks, assets, config, seed, run_dir / "editor", logger)
```

The MLP learns cosine alignment to frozen item embeddings using training-history items. The fitted MLP maps requests to localization directions. Localization averages counterfactual restoration effects across training users. The editor uses the original request vectors and selects checkpoints on the validation split.

Load the fitted MLP with `load_projection`. Use a separate run directory for each configuration. For GRU4Rec and Mamba4Rec, align to `backbone.table` and prepare grouped histories with `cler.backbone.history_vectors`. Use `protected_history` to apply compatible-history consistency correction at inference.
