## Installation

This project uses a Python virtual environment and a `pyproject.toml` configuration.

### 1. Create and activate the virtual environment

#### Windows — PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

After activation, the terminal should display `(venv)` before the prompt.

### 2. Install the dependencies

Install the dependencies listed in `requirements.txt`:

#### Windows

```powershell
python -m pip install -r requirements.txt
```

#### Linux

```bash
python3 -m pip install -r requirements.txt
```

### 3. Install the project in editable mode

Install the project using the `pyproject.toml` configuration.

#### Windows

```powershell
python -m pip install -e .
```

#### Linux

```bash
python3 -m pip install -e .
```

The `-e` option installs the project in **editable mode**, which means that changes made to the source code are immediately available without reinstalling the package.

> **Important:** Use `python -m pip` (or `python3 -m pip` on Linux) instead of simply `pip` to ensure that the package is installed into the Python environment associated with the currently active virtual environment.

### 4. Verify the installation

You can verify that the `robotics` package is correctly installed with:

#### Windows

```powershell
python -c "import robotics; print(robotics.__file__)"
```

#### Linux

```bash
python3 -c "import robotics; print(robotics.__file__)"
```

The output should point to the project's `robotics` directory, for example:

```text
.../trabalho-2/robotics/__init__.py
```

You can also test the main imports:

#### Windows

```powershell
python -c "from robotics.geometry import Pose, Angle; print('OK')"
```

#### Linux

```bash
python3 -c "from robotics.geometry import Pose, Angle; print('OK')"
```

### 5. Using the project with Jupyter or VS Code

Make sure that Jupyter or VS Code is using the Python interpreter from the project's virtual environment.

#### Windows

```text
venv\Scripts\python.exe
```

#### Linux

```text
venv/bin/python
```

To check the interpreter being used inside a notebook:

```python
import sys
print(sys.executable)
```

It should point to the Python executable inside the project's `venv`.

If the project is not recognized by the notebook after installation, reinstall it explicitly using the active environment:

#### Windows

```powershell
python -m pip install -e .
```

#### Linux

```bash
python3 -m pip install -e .
```

Then restart the Jupyter kernel.

### Development workflow

After the initial installation, you normally do **not** need to reinstall the project every time you modify the source code. Because the package is installed in editable mode, changes inside the `robotics/` directory are automatically reflected in the environment.

If the environment or installation becomes inconsistent, run:

#### Windows

```powershell
python -m pip install -e .
```

#### Linux

```bash
python3 -m pip install -e .
```
