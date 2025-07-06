# AGENTS.md

## 1. Environment

* Ensure `poetry` is installed (only once):

  ```bash
  pip install --user poetry
  ```
* Install dependencies and activate virtual environment:

  ```bash
  poetry install
  ```
* (Optional) Check dependency tree:

  ```bash
  poetry show --tree
  ```

## 2. Code Style

* Format code with **Black**:

  ```bash
  poetry run black .
  ```
* Lint with **Flake8**:

  ```bash
  poetry run flake8 .
  ```
* Type-check with **Mypy**:

  ```bash
  poetry run mypy .
  ```

## 3. Testing

* Run tests via **pytest**:

  ```bash
  poetry run pytest
  ```
* Generate coverage report (requires **coverage**):

  ```bash
  poetry run coverage run -m pytest && poetry run coverage report
  ```

## 4. PR Guidelines

* Title template:

  ```
  [module] Brief description
  ```
* Body should include:

  1. **Summary** – one-line description of change
  2. **Testing Done** – commands run and outcomes
  3. **Type Changes** – note any interface or type updates

## 5. Project Notes

* Package code lives at repository root; imports follow project name
* Tests in `tests/` matching package structure
* Configuration in `pyproject.toml` under keys:

  * `tool.poetry` for dependencies and metadata
  * `tool.black`, `tool.flake8`, `tool.mypy` for tooling
* Use `poetry run <command>` to execute commands inside the virtual environment
