'''
@file    :   config_parser.py
@create date : 2025-05-12 10:15:20
@modify date 2026-08-25 12:00:00
@author  :   Mansoor Nabawi
@version :   1.1
@contact :   mansoor.nabawi@gmail.com
@license :   (C)Copyright 2026, Physikalisch-Technische Bundesanstalt (PTB) - BioAirMet Project
@desc    :   [
    Configuration parsing, architecture resolution/merging, and experiment
    bundle helpers for BioAirMet.

    Two-layer config model
    ----------------------
    Authoring : a stage config (training knobs) + an architecture config
                (model shape: ssl_model / image_tower / fluorescence_tower).
                In stage 2 the architecture is PINNED to the SSL experiment
                bundle - it must match the SSL weights exactly and is never
                re-read from the package template.
    Runtime   : load_and_merge_architecture_config() resolves the architecture
                file and deep-merges it into architecture_setup
                (stage-config keys win over architecture-file keys).
    Archive   : experiment directories are self-contained bundles:
                resolved config.yaml + architecture.yaml + checkpoints + logs.
    Reuse     : load_experiment_config() reads a bundle. For new bundles the
                config is already merged (no-op); for old bundles the merge is
                applied on the fly (backward compatible).
    ]
'''
import os
import yaml
from easydict import EasyDict as edict


class ConfigError(RuntimeError):
    """Raised when the configuration cannot be resolved.

    The message lists every path that was searched and hints what to fix,
    so a bad config fails fast with an actionable error instead of crashing
    deep inside the model builder.
    """


def parse_config(config_path):
    """
    Parses a YAML configuration file and returns an EasyDict object.

    Args:
        config_path (str): The path to the YAML configuration file.

    Returns:
        EasyDict: An EasyDict object containing the configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return edict(config)


def deep_merge(base, override):
    """
    Deep-merge two (Easy)dicts; ``override`` wins key-by-key.

    Nested dicts are merged recursively, so a partial sub-dict override
    (e.g. only ``image_tower.model_name`` in the stage config) keeps the
    sibling defaults from the architecture file. Lists and scalars are
    replaced wholesale.
    """
    merged = edict(base or {})
    for key, value in (override or {}).items():
        current = merged.get(key)
        if isinstance(current, edict) and isinstance(value, dict):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def to_plain(obj):
    """Recursively convert EasyDicts to plain dicts/lists (for YAML dumping)."""
    if isinstance(obj, edict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def get_model_initialization(config):
    """Return the stage config's ``model_initialization`` section (or {})."""
    mi = getattr(config, 'model_initialization', None)
    return mi or {}


def get_resume_section(config):
    """Return the resume settings for the current run.

    The unified location is ``model_initialization.resume`` (used by both the
    SSL and classification stage configs). Legacy configs and old experiment
    bundles keep resume in a top-level ``resume:`` section — that location is
    used as a fallback so old configs keep working unchanged.
    """
    resume = get_model_initialization(config).get('resume') or {}
    if not resume:
        legacy = getattr(config, 'resume', None) or {}
        if legacy:
            return legacy
    return resume


def load_and_merge_architecture_config(main_config, experiment_dir=None):
    """
    Resolve and merge the architecture config for the current run.

    Resolution order (first existing file wins):
      1. ``experiment_dir/architecture.yaml`` — loading an experiment bundle
         (validation / inference / model reuse).
      2. classification + resume enabled:
         ``<dirname(model_initialization.resume.checkpoint_path)>/architecture.yaml``
         (the architecture is PINNED to the bundle the checkpoint came from).
      3. SSL + resume enabled (unified ``model_initialization.resume`` or
         legacy top-level ``resume``):
         ``<dirname(resume.checkpoint_path)>/architecture.yaml`` — the same
         self-contained rule as classification resume.
      4. classification + pretraining enabled (``pretraining.enable`` is not
         ``false``) with an ``experiment_path``:
         ``<model_initialization.pretraining.experiment_path>/architecture.yaml``
         (the architecture is PINNED to the SSL bundle so it matches the
         SSL weights exactly).
      5. otherwise (SSL / fresh training, or classification "train from
         scratch" with ``pretraining.enable: false``):
         ``architecture_setup.architecture_config_path`` if set, else the
         package folder ``config/architecture/<architecture_config_name>``,
         else ``<config file dir>/../architecture/<architecture_config_name>``
         (development checkouts).

    Precedence: stage-config ``architecture_setup.*`` keys override the
    architecture file's keys (deep-merged, so partial sub-dict overrides
    keep sibling defaults).

    The merged config is stamped with:
      ``architecture_setup.architecture_merged``  = True
      ``architecture_setup.architecture_source``  = absolute path used

    Args:
        main_config (EasyDict or dict): Main configuration object.
        experiment_dir (str, optional): Experiment bundle to load the
            architecture from (validation / inference / reuse).

    Returns:
        (main_config, arch_config_path): merged config + path of the
        architecture file actually used.

    Raises:
        ConfigError: if no architecture file can be resolved (the message
        lists every path searched and what to fix).
    """
    if not isinstance(main_config, edict):
        main_config = edict(main_config)
    if not hasattr(main_config, 'architecture_setup') or main_config.architecture_setup is None:
        main_config.architecture_setup = edict()

    arch_name = main_config.architecture_setup.get('architecture_config_name', None)
    tried = []

    def _candidate(path, reason):
        if path:
            path = os.path.abspath(os.path.expanduser(str(path)))
            tried.append((path, reason))
            return path
        return None

    arch_config_path = None

    # 1) Explicit experiment bundle (validation / inference / reuse)
    if experiment_dir:
        arch_config_path = _candidate(
            os.path.join(experiment_dir, 'architecture.yaml'),
            'experiment bundle',
        )

    # 2) + 3) classification training - architecture pinned to the referenced bundle
    if arch_config_path is None:
        training_type = main_config.architecture_setup.get('type', 'ssl')
        model_init = getattr(main_config, 'model_initialization', None)
        if training_type == 'classification' and model_init is not None:
            resume = model_init.get('resume', {}) or {}
            pretraining = model_init.get('pretraining', {}) or {}
            if resume.get('enable', False) and resume.get('checkpoint_path'):
                ckpt_dir = os.path.dirname(resume['checkpoint_path'])
                arch_config_path = _candidate(
                    os.path.join(ckpt_dir, 'architecture.yaml'),
                    'resume checkpoint directory',
                )
            elif pretraining.get('enable', True) and pretraining.get('experiment_path'):
                # Only pin to the SSL/pretraining bundle when pretraining is
                # actually enabled. When pretraining.enable is False this falls
                # through to step 4 (package template) = "train from scratch".
                arch_config_path = _candidate(
                    os.path.join(pretraining['experiment_path'], 'architecture.yaml'),
                    'SSL / pretraining experiment directory',
                )

    # 3) SSL resume - pin the architecture to the checkpoint's experiment
    #    directory (the same self-contained rule as classification resume).
    #    Works with the unified model_initialization.resume location and the
    #    legacy top-level `resume` section (old configs / bundles).
    if arch_config_path is None:
        training_type = main_config.architecture_setup.get('type', 'ssl')
        resume_cfg = get_resume_section(main_config)
        if (training_type == 'ssl' and resume_cfg.get('enable', False)
                and resume_cfg.get('checkpoint_path')):
            ckpt_dir = os.path.dirname(resume_cfg['checkpoint_path'])
            arch_config_path = _candidate(
                os.path.join(ckpt_dir, 'architecture.yaml'),
                'SSL resume checkpoint directory',
            )

    # 4) fresh (SSL) training - package folder / explicit path / dev-checkout fallback
    if arch_config_path is None:
        explicit = main_config.architecture_setup.get('architecture_config_path', None)
        if explicit:
            arch_config_path = _candidate(explicit, 'architecture_setup.architecture_config_path')
        if arch_config_path is None and arch_name:
            arch_config_path = _candidate(
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'config', 'architecture', arch_name,
                ),
                'package config/architecture folder',
            )
        if arch_config_path is None and arch_name:
            config_path = getattr(main_config, 'config_path', None)
            if config_path and os.path.isfile(config_path):
                arch_config_path = _candidate(
                    os.path.join(
                        os.path.dirname(os.path.dirname(config_path)),
                        'architecture', arch_name,
                    ),
                    'sibling config/architecture folder (development checkout)',
                )

    if arch_config_path is None or not os.path.exists(arch_config_path):
        tried_block = '\n'.join(f'    - {p}  ({r})' for p, r in tried) or '    (none - see hints below)'
        raise ConfigError(
            "Could not resolve the architecture config. Searched:\n"
            f"{tried_block}\n"
            "Hints:\n"
            "  • SSL training     - check architecture_setup.architecture_config_name "
            "(or set architecture_setup.architecture_config_path).\n"
            "  • Classification   - point model_initialization.pretraining.experiment_path "
            "at the SSL experiment directory that contains architecture.yaml, "
            "or model_initialization.resume.checkpoint_path at a valid checkpoint."
        )
    arch_config_path = os.path.abspath(arch_config_path)

    # Merge: architecture file as base, stage config on top (deep, stage wins)
    arch_config = parse_config(arch_config_path)
    merged = deep_merge(arch_config, main_config.architecture_setup)
    merged['architecture_merged'] = True
    merged['architecture_source'] = arch_config_path
    main_config.architecture_setup = merged

    return main_config, arch_config_path


def is_resolved_config(config):
    """True if the config already has the architecture merged in (new bundles)."""
    return bool(getattr(getattr(config, 'architecture_setup', None), 'architecture_merged', False))


def load_experiment_config(experiment_dir):
    """
    Load the config of a finished experiment directory (self-contained bundle).

    Works with:
      • new bundles - config.yaml already contains the merged architecture;
      • old bundles - config.yaml + architecture.yaml are merged here on the fly.

    Args:
        experiment_dir (str): Directory containing config.yaml (and, for old
            bundles, architecture.yaml) plus the checkpoints.

    Returns:
        EasyDict: fully resolved config (architecture_setup merged in).

    Raises:
        FileNotFoundError: if config.yaml is missing.
        ConfigError: if architecture.yaml is missing and the config is not
            pre-merged (incomplete bundle).
    """
    config_path = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = parse_config(config_path)
    config.config_path = config_path

    arch_path = os.path.join(experiment_dir, 'architecture.yaml')
    if os.path.exists(arch_path):
        config, _ = load_and_merge_architecture_config(config, experiment_dir=experiment_dir)
    elif not is_resolved_config(config):
        raise ConfigError(
            f"Experiment bundle at {experiment_dir} is incomplete:\n"
            "  • architecture.yaml not found, and config.yaml does not contain a merged "
            "architecture.\n"
            "  Fix: re-run the stage that produced this experiment with the current "
            "version of BioAirMet (bundles are self-contained since v1.1)."
        )

    return config


def save_resolved_config(config, dest_path):
    """
    Write the fully resolved (architecture-merged) config to ``dest_path`` as YAML.

    This is what trainers save as the experiment's config.yaml, so the
    experiment directory is a self-contained, truthful record of the run.
    """
    plain = to_plain(config)
    source = getattr(getattr(config, 'architecture_setup', None), 'architecture_source', None)
    with open(dest_path, 'w') as f:
        f.write("# BioAirMet - resolved configuration (architecture config merged in)\n")
        if source:
            f.write(f"# architecture_source: {source}\n")
        f.write(yaml.safe_dump(plain, sort_keys=False, default_flow_style=False))


def format_resolved_yaml(config):
    """Return the resolved config as a YAML string (used by ``--show-config``)."""
    return yaml.safe_dump(to_plain(config), sort_keys=False, default_flow_style=False)


def find_placeholder_paths(config, needle='/path/to'):
    """
    Return ``[(dotted_path, value), ...]`` for values that look like unfilled
    placeholders (contain ``/path/to`` or 'CHANGE ME'), used by ``--show-config``
    to build the pre-flight checklist.
    """
    hits = []

    def walk(node, prefix):
        if isinstance(node, (edict, dict)):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, (list, tuple)):
            for i, value in enumerate(node):
                walk(value, f"{prefix}[{i}]")
        elif isinstance(node, str):
            low = node.lower()
            if needle in low or 'change me' in low:
                hits.append((prefix, node))

    walk(config, '')
    return hits


def apply_overrides(config, overrides):
    """Apply command-line ``KEY=VALUE`` overrides to a (merged) config.

    Each override is a ``dotted.key=value`` string (e.g.
    ``train.epochs=5`` or ``data.val_split_ratio=0.05``). Values are parsed
    as YAML, so integers, floats, booleans, null and lists all work
    (``--set distributed.gpu_ids="[0, 1]"``). Missing intermediate sections
    are created on the fly.

    Args:
        config (EasyDict): configuration object (modified in place).
        overrides (list[str], optional): list of ``KEY=VALUE`` strings.

    Returns:
        The same config object, with the overrides applied.

    Raises:
        ConfigError: if an override is malformed.
    """
    for item in overrides or []:
        if "=" not in item:
            raise ConfigError(
                f"Invalid override {item!r}: expected 'dotted.key=value' "
                "(e.g. --set train.epochs=5)."
            )
        key, raw_value = item.split("=", 1)
        parts = [p for p in key.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"Invalid override {item!r}: empty key.")
        value = yaml.safe_load(raw_value)
        node = config
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, (dict, edict)):
                child = edict()
                node[part] = child
            node = child
        node[parts[-1]] = value
    return config


def validate_config(config, mode):
    """Fail fast on missing or invalid required settings before training starts.

    Complements the placeholder check: this verifies the *shape* of the
    config (required sections, numeric ranges, mutual exclusivity) so the
    user gets one actionable error listing ALL problems instead of a crash
    deep inside a worker process.

    Args:
        config (EasyDict): fully merged configuration object.
        mode (str): training mode — 'ssl' or 'classification'.

    Raises:
        ConfigError: listing every problem found.
    """
    problems = []

    def _sec(name):
        val = config.get(name)
        if val is None:
            problems.append(f"missing section: {name!r}")
            return edict()
        return val

    data = _sec("data")
    train = _sec("train")
    logging = _sec("logging")

    # logging.checkpoint_dir — required in every mode
    cdir = logging.get("checkpoint_dir")
    if not isinstance(cdir, str) or not cdir.strip():
        problems.append("logging.checkpoint_dir is required (non-empty string)")

    # train basics — required in every training mode
    for key, minimum in (("epochs", 1), ("batch_size", 1), ("num_workers", 0)):
        val = train.get(key)
        if not isinstance(val, int) or isinstance(val, bool) or val < minimum:
            problems.append(f"train.{key} must be an integer >= {minimum}, got {val!r}")

    if mode == "ssl":
        dp = data.get("dataset_path")
        if not isinstance(dp, str) or not dp.strip():
            problems.append("data.dataset_path is required for SSL training")
        ratio = data.get("val_split_ratio")
        if ratio is not None and (
            not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
            or not 0.0 <= ratio < 1.0
        ):
            problems.append("data.val_split_ratio must be a number in [0, 1)")

    elif mode == "classification":
        for key in ("train_data_path", "validation_path", "cat_map_path"):
            if not data.get(key):
                problems.append(f"data.{key} is required for classification training")
        arch_setup = config.get("architecture_setup") or {}
        cm = arch_setup.get("classification_model") or {}
        nc = cm.get("num_classes")
        if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
            problems.append(
                "architecture_setup.classification_model.num_classes must be "
                f"an integer >= 1, got {nc!r}"
            )
        pre = get_model_initialization(config).get("pretraining") or {}
        if pre.get("enable", True) and not pre.get("experiment_path"):
            problems.append(
                "model_initialization.pretraining.experiment_path is required "
                "when pretraining.enable is true (or set enable: false to train from scratch)"
            )

    # mutual exclusion: warm-start (pretrained) vs resuming
    mi = get_model_initialization(config)
    resume = mi.get("resume") or {}
    if not resume:
        resume = getattr(config, "resume", None) or {}
    pretrained = mi.get("pretrained") or {}
    if not pretrained:
        pretrained = ((config.get("architecture_setup") or {}).get("pretrained")) or {}
    if (pretrained.get("enable") or pretrained.get("enabled")) and resume.get("enable"):
        problems.append(
            "model_initialization.pretrained.enable and model_initialization.resume.enable "
            "are both true — enable at most one"
        )

    if problems:
        raise ConfigError(
            "Configuration is invalid:\n  - " + "\n  - ".join(problems)
        )


def resolve_inference_settings(config):
    """
    Resolve the inference/evaluation batch size and DataLoader worker count
    from a merged config.

    The canonical location is the ``train:`` section (where the shipped
    templates put them); ``data:`` is accepted as a fallback for
    non-standard configs. An explicit ``0`` is respected (e.g.
    ``num_workers: 0`` → no DataLoader workers); only *missing* or invalid
    values fall back to defaults.

    Args:
        config: EasyDict (or dict) merged configuration object.

    Returns:
        (batch_size, batch_source, num_workers, workers_source)
        where each *_source is one of 'train', 'data', 'default'.
    """
    def _get(section, key):
        if isinstance(config, dict):
            sec = config.get(section)
        else:
            sec = getattr(config, section, None)
        if sec is None:
            return None, False
        try:
            val = int(sec[key])
        except (KeyError, TypeError, ValueError, IndexError):
            return None, False
        return (val, True) if val >= 0 else (None, False)

    def _pick(key, default):
        for section in ("train", "data"):
            val, present = _get(section, key)
            if present:
                return val, section
        return default, "default"

    batch_size, batch_source = _pick("batch_size", 64)
    num_workers, workers_source = _pick("num_workers", 6)
    return batch_size, batch_source, num_workers, workers_source
