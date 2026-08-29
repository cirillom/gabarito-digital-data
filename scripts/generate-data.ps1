param(
    [switch]$RegenerateAll,
    [switch]$SkipRichContent,
    [switch]$UseGeminiRich,
    [ValidateSet('gemini', 'openai')]
    [string]$RichProvider,
    [string]$RichModel,
    [int[]]$Question,
    [string]$PreviewDir
)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repositoryRoot
try {
    if ($UseGeminiRich -and $RichProvider -and $RichProvider -ne 'gemini') {
        throw '-UseGeminiRich cannot be combined with -RichProvider openai.'
    }
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
    $selectedProvider = if ($RichProvider) { $RichProvider } elseif ($UseGeminiRich) { 'gemini' }
    if ($selectedProvider) {
        $arguments += @('--rich-provider', $selectedProvider)
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
    if (-not $SkipRichContent -and -not $Question) {
        uv run main.py --check-rich-content
    }
}
finally {
    Pop-Location
}
