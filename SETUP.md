### Project setup with `uv` and Jupyter

Follow these steps whenever you clone or create a new project that uses `uv` and Jupyter notebooks.

#### 1. Go to the project directory

```bash
cd /path/to/my-project
```

#### 2. Create (or recreate) the `uv` virtual environment

```bash
uv venv
```

If you suspect the env is broken or paths changed, delete it first:

```bash
rm -rf .venv
uv venv
```

#### 3. Install project dependencies

If you have `pyproject.toml` / `uv.lock`:

```bash
uv sync
```

Or if you use `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

#### 4. Install Jupyter and `ipykernel` into the env

```bash
uv pip install jupyter ipykernel
```

#### 5. Register a Jupyter kernel for this project

Use a unique name per project (change `my-project-uv` as needed):

```bash
KERNEL_NAME="my-project-uv"
DISPLAY_NAME="Python (my-project-uv)"

uv run python -m ipykernel install --user \
  --name "${KERNEL_NAME}" \
  --display-name "${DISPLAY_NAME}"
```

#### 6. Select the kernel in your notebook

- Open the `.ipynb` file.
- Use **Select Kernel**.
- Choose the display name you configured (for example: `Python (my-project-uv)`).

#### 7. Quick troubleshooting

- **Check which Python `uv` is using**

  ```bash
  uv run which python
  ```

  It should point inside `.venv` in your project.

- **List available kernels**

  ```bash
  uv run python -m jupyter kernelspec list
  ```

  Make sure your kernel name appears.

- **If paths changed or venv is corrupted**

  ```bash
  rm -rf .venv
  uv venv
  uv sync          # or: uv pip install -r requirements.txt
  uv pip install jupyter ipykernel
  uv run python -m ipykernel install --user --name my-project-uv --display-name "Python (my-project-uv)"
  ```

