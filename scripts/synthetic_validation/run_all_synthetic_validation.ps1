param(
    [switch]$Full,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"

if (-not $SkipTests) {
    Write-Host "[1/4] Running test suite"
    python -m pytest -q
}

$PairResults = "outputs\synthetic_validation\pair_family_recovery_full\pair_family_recovery_results.csv"
if (-not (Test-Path $PairResults)) {
    $PairResults = "pair_family_recovery_full\pair_family_recovery_results.csv"
}
if (-not (Test-Path $PairResults)) {
    throw "Could not find the completed pair-family results CSV."
}

Write-Host "[2/4] Creating pair-family report"
python scripts\synthetic_validation\plot_pair_recovery_results.py `
    --results $PairResults `
    --output-directory reports\synthetic_validation\pair_family_recovery

if ($Full) {
    $Config = "configs\synthetic_validation\complete_synthetic_validation_full.yaml"
    $ResultsDirectory = "outputs\synthetic_validation\complete_full"
    $ReportDirectory = "reports\synthetic_validation\complete_full"
} else {
    $Config = "configs\synthetic_validation\complete_synthetic_validation_compact.yaml"
    $ResultsDirectory = "outputs\synthetic_validation\complete_compact"
    $ReportDirectory = "reports\synthetic_validation\complete_compact"
}

Write-Host "[3/4] Running remaining synthetic validation"
python scripts\synthetic_validation\run_complete_synthetic_validation.py `
    --config $Config `
    --overwrite

Write-Host "[4/4] Creating complete report"
python scripts\synthetic_validation\plot_complete_synthetic_validation.py `
    --results-directory $ResultsDirectory `
    --report-directory $ReportDirectory

Write-Host "Synthetic validation complete."
Write-Host "Pair report:     reports\synthetic_validation\pair_family_recovery\README.md"
Write-Host "Complete report: $ReportDirectory\README.md"
