param(
    [switch]$SkipTests,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"

if (-not $SkipTests) {
    python -m pytest -q
}

$datasets = @(
    @{
        Config = "configs/datasets/sachs.yaml"
        Summary = "reports/tables/sachs/conditional_vine_summary_indegree2.csv"
        Report = "reports/conditional_models/sachs"
    },
    @{
        Config = "configs/datasets/diabetes.yaml"
        Summary = "reports/tables/diabetes/conditional_vine_summary_indegree2.csv"
        Report = "reports/conditional_models/diabetes"
    },
    @{
        Config = "configs/datasets/breast_cancer.yaml"
        Summary = "reports/tables/breast_cancer/conditional_vine_summary_indegree2.csv"
        Report = "reports/conditional_models/breast_cancer"
    }
)

foreach ($dataset in $datasets) {
    $arguments = @(
        "scripts/build_conditional_vine_library.py",
        "--config", $dataset.Config
    )
    if ($Limit -gt 0) {
        $arguments += @("--limit", "$Limit")
    }

    python @arguments

    python scripts/report_conditional_model_selection.py `
        --summary $dataset.Summary `
        --output-dir $dataset.Report
}

Write-Host "Adaptive conditional-vine rebuild complete."
