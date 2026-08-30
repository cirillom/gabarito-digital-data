param(
    [switch]$RegenerateAll,
    [switch]$RichContent,
    [switch]$GeminiRich,
    [string]$RichModel,
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
    if ($RichContent) {
        $arguments += '--rich-content'
    }
    if ($GeminiRich) {
        $arguments += '--gemini-rich'
    }
    if ($RichModel) {
        $arguments += @('--rich-model', $RichModel)
    }
    foreach ($number in $Question) {
        $arguments += @('--rich-question', $number)
    }
    if ($PreviewDir) {
        $arguments += @('--preview-dir', $PreviewDir)
    }
    uv run @arguments
    uv run main.py --validate
}
finally {
    Pop-Location
}
