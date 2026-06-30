# gravityml

A configurable toolkit for certifying tabular datasets and training regression
models on them. Data is certified into a typed domain at the edge and persisted as
Parquet under a configurable state directory; models carry their own training
binding, so the lifecycle is `prepare -> define -> train -> evaluate -> predict`.

The dataset schema (which columns a source must carry) is CONFIGURATION, not code,
so the same tool serves any tabular dataset. No real-world data ships with the
project -- only a small synthetic example so the quickstart runs out of the box.

## Requirements

- Python 3.14+ (the version is pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/) for environment and dependency management

## Install

```bash
make install
```

This builds the local virtual environment and installs the project (editable)
into it, exposing the `gravityml` command. Equivalent to `uv sync`.

## Quickstart

The bundled synthetic example (`data/example/orig/example.csv`, columns `x1`,
`x2`, `y`) matches the DEFAULT schema, so this runs with no configuration:

```bash
# 1. Install (editable, into .venv)
make install

# 2. Prepare a dataset: certify a source file and persist it as Parquet
#    -> <state>/datasets/example.parquet
uv run gravityml prepare-dataset example -i data/example/orig/example.csv

# 3. Describe it: write the univariate-statistics table
#    -> <state>/reports/example.univariate-stats.csv
uv run gravityml describe-dataset example

# 4. Define + train a small model, then predict
uv run gravityml define-model demo \
  --input-dim 2 --target-dim 1 --dataset example \
  --feature-cols x1,x2 --target-cols y \
  --hidden-dims 16 --scaler standard --max-epochs 30 --batch-size 32
uv run gravityml train-model demo
uv run gravityml predict demo -i data/example/orig/example.csv
```

Each step is detailed below; every path is configurable (see Configuration).

## Using your own data

Nothing about the tool is tied to the example. To run it on your own data:

1. Put a `.csv` or `.parquet` file anywhere (a natural home is
   `data/<name>/orig/` -- see `data/README.md`).
2. Declare the columns it must carry, as floats, via `required_columns` (env or
   TOML -- see Configuration). Any other source columns are dropped on certify.
3. `uv run gravityml prepare-dataset <name> -i path/to/your/source.csv`.

## Configuration

Settings resolve from a `gravityml.toml` file in the project root, layered UNDER
`GRAVITYML__`-prefixed environment variables -- **env always wins**. Nested keys
use the same `__` delimiter (e.g. `GRAVITYML__TRAIN__MAX_EPOCHS`). Every setting
has a default, so configuration is optional.

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| `state` | `GRAVITYML__STATE` | `state` | Root directory for all runtime artifacts |
| `required_columns` | `GRAVITYML__REQUIRED_COLUMNS` | `["x1","x2","y"]` | Columns a source MUST carry (as floats) to certify |
| `datasets_dir` | (derived) | `<state>/datasets` | Where prepared datasets are written |
| `reports_dir` | (derived) | `<state>/reports` | Where analytical outputs are written |
| `models_dir` | (derived) | `<state>/models` | Where model manifests and trained weights are written |

Example `gravityml.toml`:

```toml
state = "state"
required_columns = ["x1", "x2", "y"]
```

`required_columns` from the environment is JSON (env wins over TOML):

```bash
# fish
set -x GRAVITYML__REQUIRED_COLUMNS '["feature_a", "feature_b", "target"]'
# bash / zsh
export GRAVITYML__REQUIRED_COLUMNS='["feature_a", "feature_b", "target"]'
```

## Data and artifacts

| Location | Version-controlled? | Contents |
|---|---|---|
| `data/<name>/orig/` | Your choice | Original source data (the inputs you start from) |
| `data/<name>/prepared/` | No (gitignored) | Intermediate prepared data -- a generated artifact |
| `state/datasets/` | No (gitignored) | Prepared datasets (`<id>.parquet`) |
| `state/reports/` | No (gitignored) | Analytical outputs (e.g. `<id>.univariate-stats.csv`) |
| `state/models/` | No (gitignored) | Model manifests (`<id>.json`) + trained weights (`<id>.safetensors`) |

Everything generated under `state/` (and `data/<name>/prepared/`) is ignored and
reproducible from a source plus a command. The only data committed to this repo is
the synthetic `data/example/orig/example.csv`, regenerable with
`uv run python scripts/data/make_example_data.py`.

## Usage

See the available commands:

```bash
uv run gravityml --help
```

## Preparing a dataset

The first step in any workflow is to prepare a dataset: gravityml reads your
source file, certifies it against the configured schema, and persists it as
Parquet (which preserves column dtypes and embeds the dataset identity).

```bash
uv run gravityml prepare-dataset <name> -i path/to/source.csv
```

- The source (`-i` / `--input`) may be `.csv` or `.parquet` and must carry the
  configured `required_columns`, each of a float dtype.
- On success the certified dataset is written to `<state>/datasets/<name>.parquet`
  (per your configuration) and the path is printed. The persisted frame holds only
  the required columns -- any extra source columns are dropped.
- If the source fails certification (missing or non-float columns) the command
  reports the reason and exits non-zero, writing nothing.

## Describing a dataset

Tabulate univariate statistics for a prepared dataset -- per column: `n_unique`,
min, max, mean, median, std, the 1/5/25/50/75/95/99 percentiles, skewness, and
kurtosis.

```bash
uv run gravityml describe-dataset <name>
```

- Reads `<state>/datasets/<name>.parquet` and writes the table to
  `<state>/reports/<name>.univariate-stats.csv` (the filename states the dataset),
  printing the saved path.
- Conventions: sample std (`ddof=1`), Fisher-Pearson sample skewness, Fisher
  (excess) kurtosis (normal -> 0), `median` == `p50`.
- If the dataset has not been prepared yet, the command reports the read failure
  and exits non-zero.

## Defining a model

Define a training-ready model: certify its recipe (network, optimizer, loss) AND
its training binding (which dataset, the feature/target columns, and the run
settings), then persist it. Because the model carries its own binding,
`train-model` later needs only the id.

```bash
uv run gravityml define-model <model_id> \
  --input-dim 2 --target-dim 1 \
  --dataset <name> \
  --feature-cols x1,x2 --target-cols y \
  --max-epochs 50 --batch-size 64
```

- Recipe flags default to a sequential-MLP regression: `--network`
  (`sequential-mlp`), `--hidden-dims` (comma-separated widths, default none = a
  linear map), `--activation` (`relu`), `--scaler` (`identity`; `standard`
  standardizes features), `--optimizer` (`adam`), `--learning-rate` (`0.001`),
  `--weight-decay` (`0.0`), `--loss` (`mse`). `--input-dim` and `--target-dim` are
  required and must match the feature / target counts.
- `--scaler` chooses how features enter the network: `identity` (default) passes
  them through raw, while `standard` standardizes each feature to
  `(x - mean) / std`. The per-feature statistics are fitted at `train-model` time
  and stored inside the `.safetensors` weights, so `predict` reapplies them to a
  raw input file. Use `standard` when features span very different scales (a large
  raw feature can otherwise dominate the first layer and make the loss diverge).
- Binding flags `--dataset`, `--feature-cols`, and `--target-cols` (comma-separated
  -- a single target is one name) are required; `--val-fraction` (default `0.0`),
  `--seed` (default `0`), and `--accelerator` (`cpu` / `gpu` / `auto`, default
  `auto`) tune the run.
- On success the model is persisted to `<state>/models/<model_id>.json` and the
  path is printed. Invalid hyperparameters (non-positive dimensions, learning
  rate, epochs / batch, or a `val_fraction` outside `[0, 1)`) report the reason
  and exit non-zero, writing nothing.

## Training a model

Train a fully-defined model. The data binding and run settings were fixed at
define time, so training takes only the model id.

```bash
uv run gravityml train-model <model_id>
```

- Loads the defined model, reads its bound dataset from `<state>/datasets/`, runs
  the fit, and writes the trained weights to
  `<state>/models/<model_id>.safetensors` (pickle-free) alongside the updated
  `<model_id>.json` manifest; the path is printed.
- A model that does not exist (or is not in the `defined` state) reports the
  failure and exits non-zero.

## Evaluating a model

Evaluate a trained model on a stored (test) dataset, recording its test loss.

```bash
uv run gravityml evaluate-model <model_id> \
  --dataset <name>-test \
  --feature-cols x1,x2 --target-cols y
```

- Loads the trained model and its weights, runs a forward pass over the WHOLE
  referenced dataset (no validation split), and records `test_loss` and
  `n_samples` into the manifest; the path is printed.
- A model that is not trained, or a missing dataset, reports the failure and exits
  non-zero.

## Predicting with a model

Run a trained model over an input file and write its target predictions.

```bash
uv run gravityml predict <model_id> -i path/to/inputs.csv
```

- The input (`-i` / `--input`, `.csv` or `.parquet`) must carry the model's
  feature columns; predict reads them straight off the stored model.
- Writes the predicted target columns as CSV next to the input, at
  `<input>.predictions.csv`, and prints the path.
- A model that is not trained, a missing input file, or absent feature columns
  reports the failure and exits non-zero.

## Describing a model

Report a stored model's lifecycle state and hyperparameters.

```bash
uv run gravityml describe-model <model_id>
```

- Reads `<state>/models/<model_id>.json` and prints, to stdout, the model's
  `status` (`defined` / `trained` / `evaluated` / `archived`) and its recipe:
  the `network` (input/hidden/target dims, activation), the `optimizer`
  (learning rate, weight decay), and the `loss`.
- Once trained, an extra `training` line reports the data binding (`dataset`,
  `features`, `targets`), the split (`val_fraction`), the run settings, and the
  metrics. Once evaluated, an `evaluation` line adds `test_loss`, `n_samples`, and
  `evaluated_at`. An archived model also prints its `archived_at` time.
- This is a read-only query: it inspects the stored manifest and never mutates it.

## Listing models

List every stored model with its lifecycle status.

```bash
uv run gravityml list-models
```

- Reads every `<state>/models/<id>.json` manifest and prints one `<id>: <status>`
  line per model (sorted by id), or `no models stored` when the store is empty.
- This is a read-only query; it never mutates the store.

## Archiving a model

Retire a stored model -- tombstone whatever live state it is in (`defined`,
`trained`, or `evaluated`). The tombstone preserves the full prior record and is
terminal: an archived model cannot be archived again.

```bash
uv run gravityml archive-model <model_id>
```

- Loads whichever live state is stored, records the archival time, and persists
  the tombstone to `<state>/models/<model_id>.json`; the path is printed.
- A model that does not exist, or one already archived, reports the failure and
  exits non-zero.

## Development

```bash
make test          # pytest with coverage
make typecheck     # mypy
make lint          # ruff + import-linter contracts
```

The architecture is a functional core / imperative shell with onion rings per
bounded context (`datasets`, `mlmodel`, `cli`); the import boundaries are enforced
by `.importlinter` (run via `make lint`).
