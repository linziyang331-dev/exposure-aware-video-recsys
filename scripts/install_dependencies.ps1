# Install additions to the D: project, without modifying global packages.
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'enter_env.ps1')
$recsysRoot = Split-Path -Parent $PSScriptRoot
$recsysLogDir = Join-Path $recsysRoot 'reports\environment_logs'
New-Item -ItemType Directory -Path $recsysLogDir -Force | Out-Null
$recsysPreviousRequireVenv = $env:PIP_REQUIRE_VIRTUALENV
try {
    $env:PIP_REQUIRE_VIRTUALENV = 'false'
    & $env:RECSYS_PYTHON -B -m pip install --index-url https://pypi.org/simple --only-binary=:all: --no-compile --no-deps --target $env:RECSYS_PACKAGE_DIR --requirement (Join-Path $recsysRoot 'requirements\main.txt') --report (Join-Path $recsysLogDir 'install_python312.json') --log (Join-Path $recsysLogDir 'install_python312.log')
    if ($LASTEXITCODE -ne 0) { throw "Dependency install failed: exit $LASTEXITCODE" }
    & $env:RECSYS_PYTHON -B (Join-Path $PSScriptRoot 'apply_recbole_patch.py')
    if ($LASTEXITCODE -ne 0) { throw 'RecBole compatibility patch failed' }
} finally {
    $env:PIP_REQUIRE_VIRTUALENV = $recsysPreviousRequireVenv
}
