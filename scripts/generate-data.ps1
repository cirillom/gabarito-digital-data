param(
    [switch]$RegenerateAll,
    [switch]$SkipRichContent,
    [switch]$UseGeminiRich,
    [int[]]$Question,
    [string]$PreviewDir
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repositoryRoot
try {
    uv sync --locked
    uv run python -m unittest discover -v
    $arguments = @('main.py')
    if ($RegenerateAll) {
        $arguments += '--regenerate-all'
    }
    if (-not $SkipRichContent) {
        $arguments += '--rich-content'
    } else {
        $arguments += '--no-rich-content'
    }
    if ($UseGeminiRich) {
        $arguments += '--use-gemini-rich'
    }
    foreach ($number in $Question) {
        $arguments += @('--rich-question', $number)
    }
    if ($PreviewDir) {
        $arguments += @('--preview-dir', $PreviewDir)
    }
    uv run @arguments
    if (-not $SkipRichContent -and -not $Question) {
        uv run main.py --check-rich-content
    }
}
finally {
    Pop-Location
}
