Temporarily allow script execution for this session
Run this command in your PowerShell before activating your venv:

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process


Then activate your venv:

venv\Scripts\activate.bat

---------------------------------------------------------------------------------
python -m pip install --upgrade pip

----------------------------------------

Install the required packages


pip install numpy==1.24.3 onnx==1.14.1 onnxruntime==1.15.1 torch torchvision
-----------------------------------------

Verify your environment


python --version
python -c "import numpy; print(numpy.__version__)"

You should see Python 3.10.x and numpy 1.24.3.

---------------------------------------

python loop_single_instance.py

----------------------------------------------------------------------------------

- Always activate your venv before running your scripts.
- Use `python`, not `py`, after activation to ensure you use the venv’s interpreter.

