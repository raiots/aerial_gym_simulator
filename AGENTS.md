# Repository Guidelines

## Project Structure & Module Organization
- `aerial_gym/` — main Python package.
  - `config/` (env/robot/task/controller/sim configs), `control/`, `env_manager/`, `robots/`, `sensors/`, `sim/`, `task/`, `utils/`.
  - `examples/` — runnable scripts; good for smoke tests and demos.
  - `rl_training/` — RL integrations (rl_games, CleanRL) and YAML configs.
- `resources/` — assets and ancillary files.
- `docs/` + `mkdocs.yml` — documentation site (MkDocs Material).
- `requirements.txt`, `setup.py`, `pyproject.toml` — packaging and tooling.
- Ignored outputs: `runs/`, `nn/`, `wandb/`, `*.tfevents*`, `*.pt` (see `.gitignore`).

## Build, Test, and Development Commands
- Use the venv python (expected to replace python command at everytime running): `export LD_LIBRARY_PATH=~/anaconda3/envs/aerialgym/lib && /home/raiot/anaconda3/envs/aerialgym/bin/python`
- Run example (headless): `python aerial_gym/examples/position_control_example.py --headless True --num_envs 64`
- Train RL (rl_games): `python aerial_gym/rl_training/rl_games/runner.py --train --file aerial_gym/rl_training/rl_games/ppo_aerial_quad.yaml --task navigation_task --headless True`
- Docs (local preview): `pip install mkdocs-material && mkdocs serve`

## Coding Style & Naming Conventions
- Python 3; 4‑space indents; max line length 100 (Black configured in `pyproject.toml`).
- Run formatter: `black .`
- Naming: modules/files `lower_snake_case`; functions/vars `snake_case`; classes `PascalCase`.
- Prefer `aerial_gym.utils.logging.CustomLogger` over bare prints.
- Keep configs small and composable under `aerial_gym/config/*`.

## Testing Guidelines
- No formal test suite yet. Use example scripts as smoke tests (prefer `--headless True` and small `--num_envs`).
- If adding tests, place under `tests/`, use `pytest`, and skip GPU/Isaac‑Gym heavy tests when unavailable.

## Commit & Pull Request Guidelines
- Use Conventional Commit style where possible: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`.
- Message: imperative, concise, include scope when helpful (e.g., `feat(task): add sim2real variant`).
- PRs must include: clear description, linked issues (e.g., `Closes #123`), run instructions, and screenshots/GIFs for visual changes.
- Do not commit generated artifacts (`runs/`, `wandb/`, `nn/`, large binaries). Keep diffs minimal and focused.

## Security & Configuration Tips
- Do not hardcode absolute paths, credentials, or API keys. Use repo‑relative paths and configs in `aerial_gym/config/*`.
- Large assets belong in `resources/` or external storage; reference them via config.
- Prefer headless runs for CI/review; use `--use_warp True` when appropriate.

## Agent‑Specific Notes
- Keep changes surgical; match existing structure and style. Update docs/examples when adding new tasks/controllers.
- Avoid broad refactors without discussion; prefer incremental PRs with runnable examples.

