$ErrorActionPreference = "Stop"

param(
  [Parameter(Mandatory = $true)]
  [string]$Csv,
  [Parameter(Mandatory = $true)]
  [string]$Out
)

python -m thinkrouter.experiments.summarize $Csv --out $Out
