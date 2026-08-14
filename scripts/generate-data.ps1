param(
    [switch]$RegenerateAll
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repositoryRoot
try {
    uv sync --locked
    uv run python -m unittest discover -v
    if ($RegenerateAll) {
        uv run main.py --regenerate-all
    }
    else {
        uv run main.py
    }
}
finally {
    Pop-Location
}
