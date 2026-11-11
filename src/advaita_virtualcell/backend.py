"""PIE integration with explicit, relocatable asset configuration."""

import contextlib
import os
from pathlib import Path

import numpy as np

from .io import read_json, resolve_path


def require_pie():
    try:
        import pie  # noqa: F401
        import torch  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Model execution requires the 'model' extra and PIE assets; see docs/assets.md"
        ) from error


def configure_assets(path):
    path = Path(path).resolve()
    values = read_json(path)
    root = path.parent
    for key, variable in (
        ("cache_dir", "PIE_CACHE_DIR"),
        ("data_root", "PIE_DATA_ROOT"),
        ("runs_root", "PIE_RUNS_ROOT"),
    ):
        if key not in values:
            raise ValueError(f"Asset configuration is missing {key}")
        destination = resolve_path(values[key], root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(destination)
    required = {"preprocessed_dirs", "source_dirs", "gene_text_dir", "split_dir"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"Asset configuration is missing: {sorted(missing)}")

    def local(value):
        if value.startswith("hf://"):
            return value
        resolved = resolve_path(value, root).resolve()
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return str(resolved)

    data = {
        "preprocessed_dirs": [local(p) for p in values["preprocessed_dirs"]],
        "source_dirs": {key: local(p) for key, p in values["source_dirs"].items()},
        "gene_text_dir": local(values["gene_text_dir"]),
        "split_dir": local(values["split_dir"]),
        "num_workers": 0,
    }
    if "aliases_path" in values:
        data["aliases_path"] = local(values["aliases_path"]) if values["aliases_path"] else None
    return data


def load_model_checkpoint(path, data_overrides):
    require_pie()
    from pie.predict import load_checkpoint

    checkpoint = load_checkpoint(Path(path))
    data = checkpoint.config.data.model_copy(update=data_overrides)
    checkpoint.config = checkpoint.config.model_copy(update={"data": data})
    return checkpoint


def select_device(value="auto"):
    import torch

    device = torch.device(
        "cuda"
        if value == "auto" and torch.cuda.is_available()
        else "cpu"
        if value == "auto"
        else value
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def autocast(device, precision="auto"):
    import torch

    if precision not in {"auto", "float32", "bfloat16"}:
        raise ValueError("precision must be auto, float32, or bfloat16")
    if precision == "bfloat16" or (precision == "auto" and device.type == "cuda"):
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def subset_sample(sample, indices, token_limit=128, *, include_context_text=False):
    import torch

    result = dict(sample)
    selected = torch.as_tensor(indices, dtype=torch.long)
    for key in (
        "gene_ids",
        "ctrl_mean",
        "fold_changes",
        "de_mask",
        "tested",
        "delta_p",
        "lfc_true",
        "precision",
        "lfc_label",
    ):
        if key in sample:
            result[key] = sample[key][selected]
    result["evidence"] = {key: value[selected] for key, value in sample["evidence"].items()}
    for key in ("target_gene", "target_gene_idx"):
        found = np.flatnonzero(np.asarray(indices) == sample[key])
        result[key] = int(found[0]) if len(found) else -1
    result["source_tokens"] = {
        key: value[:token_limit]
        for key, value in sample["source_tokens"].items()
        if include_context_text or key != "context_text"
    }
    return result


def construct_model(checkpoint, datamodule, device):
    from pie.model import PieModel

    model = PieModel(
        checkpoint.config.model,
        checkpoint.stats,
        datamodule.source_dims,
        datamodule.gene_query_text(),
    )
    model.load_state_dict(checkpoint.state_dict, strict=True)
    return model.to(device).eval()
