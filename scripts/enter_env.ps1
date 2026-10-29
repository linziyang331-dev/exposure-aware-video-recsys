# Dot-source: . .\scripts\enter_env.ps1
# Existing Python 3.12/PyTorch; no environment creation or installation.
$ErrorActionPreference = 'Stop'
$recsysProjectRoot = Split-Path -Parent $PSScriptRoot
$recsysPython = (Get-Command python.exe -ErrorAction Stop).Source
& $recsysPython -B -c "import sys; assert sys.version_info[:2] == (3, 12), 'Select the existing Python 3.12 interpreter'"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required' }
$recsysRuntimeRoot = Join-Path $recsysProjectRoot '.runtime'
$recsysPackages = Join-Path $recsysRuntimeRoot 'python312-packages'
foreach ($recsysDirectory in @('tmp','pip-cache','python-userbase','torch','huggingface','cache','matplotlib','numba','ipython','jupyter-config','jupyter-data','jupyter-runtime','wandb','torchinductor','triton','python312-packages')) {
    New-Item -ItemType Directory -Path (Join-Path $recsysRuntimeRoot $recsysDirectory) -Force | Out-Null
}
$env:RECSYS_PYTHON = $recsysPython
$env:RECSYS_PACKAGE_DIR = $recsysPackages
$env:PYTHONPATH = "$recsysPackages;$recsysProjectRoot"
$env:TEMP = Join-Path $recsysRuntimeRoot 'tmp'
$env:TMP = $env:TEMP
$env:PIP_CACHE_DIR = Join-Path $recsysRuntimeRoot 'pip-cache'
$env:PIP_CONFIG_FILE = 'NUL'
$env:PIP_REQUIRE_VIRTUALENV = 'true'
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$env:PYTHONUSERBASE = Join-Path $recsysRuntimeRoot 'python-userbase'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:TORCH_HOME = Join-Path $recsysRuntimeRoot 'torch'
$env:HF_HOME = Join-Path $recsysRuntimeRoot 'huggingface'
$env:XDG_CACHE_HOME = Join-Path $recsysRuntimeRoot 'cache'
$env:MPLCONFIGDIR = Join-Path $recsysRuntimeRoot 'matplotlib'
$env:NUMBA_CACHE_DIR = Join-Path $recsysRuntimeRoot 'numba'
$env:IPYTHONDIR = Join-Path $recsysRuntimeRoot 'ipython'
$env:JUPYTER_CONFIG_DIR = Join-Path $recsysRuntimeRoot 'jupyter-config'
$env:JUPYTER_DATA_DIR = Join-Path $recsysRuntimeRoot 'jupyter-data'
$env:JUPYTER_RUNTIME_DIR = Join-Path $recsysRuntimeRoot 'jupyter-runtime'
$env:WANDB_DIR = Join-Path $recsysRuntimeRoot 'wandb'
$env:WANDB_CACHE_DIR = $env:WANDB_DIR
$env:TORCHINDUCTOR_CACHE_DIR = Join-Path $recsysRuntimeRoot 'torchinductor'
$env:TRITON_CACHE_DIR = Join-Path $recsysRuntimeRoot 'triton'
Remove-Item Env:VIRTUAL_ENV,Env:PIP_TARGET,Env:PIP_PREFIX,Env:PIP_USER -ErrorAction SilentlyContinue
Write-Host "Existing Python: $recsysPython"
Write-Host "Project dependencies: $recsysPackages"
