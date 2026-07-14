param(
    [switch]$SkipTests,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

if (-not $SkipTests) {
    python -m pytest -q
}

$configs = @(
    "configs/datasets/sachs.yaml",
    "configs/datasets/diabetes.yaml",
    "configs/datasets/breast_cancer.yaml"
)

foreach ($config in $configs) {
    $arguments = @(
        "scripts/build_conditional_vine_library.py",
        "--config", $config,
        "--parent-set-size", "3"
    )

    if ($Limit -gt 0) {
        $arguments += @("--limit", "$Limit")
    }

    Write-Host "============================================================"
    Write-Host "Building indegree-three library for $config"
    Write-Host "============================================================"
    python @arguments
}

Write-Host "Indegree-three library rebuild complete."
