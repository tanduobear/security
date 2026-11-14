param(
    [ValidateSet('Check','M1','M2Train','M2Evaluate','M3Search','M3Evaluate')]
    [string]$Stage = 'Check',
    [string]$RunRoot = 'C:\Users\Administrator\Desktop\Security\p7-run-3632afa',
    [string]$Python = 'C:\Users\Administrator\Desktop\Security\repo\.venv-p6\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
$RunRoot = (Resolve-Path -LiteralPath $RunRoot).Path
$Python = (Resolve-Path -LiteralPath $Python).Path
Set-Location -LiteralPath $RunRoot
# Trust only this explicitly selected worktree for this process and its children.
# Worktrees created by the sandbox may have a different Windows owner.
if ($RunRoot.Contains("'")) { throw 'Choose a run directory without single quotes.' }
$gitTrust = "'safe.directory=" + $RunRoot.Replace('\','/') + "'"
$env:GIT_CONFIG_PARAMETERS = ($env:GIT_CONFIG_PARAMETERS + ' ' + $gitTrust).Trim()
$requiredCommit = (& git rev-parse '3632afa^{commit}').Trim()
$actualCommit = (& git rev-parse HEAD).Trim()
if ($actualCommit -ne $requiredCommit) { throw 'Run checkout must remain exactly at 3632afa.' }
& git diff --quiet HEAD --
if ($LASTEXITCODE -ne 0) { throw 'Tracked source/data changes detected. Restore the approved checkout before training.' }
$receiptRoot = Join-Path $RunRoot 'results\p7_corrected\receipts'
$logRoot = Join-Path $RunRoot 'results\p7_corrected\logs'
New-Item -ItemType Directory -Force -Path $receiptRoot,$logRoot | Out-Null
$receiptPath = Join-Path $receiptRoot ($Stage + '.json')
if ($Stage -ne 'Check') {
    $checkPath = Join-Path $receiptRoot 'Check.json'
    if (-not (Test-Path -LiteralPath $checkPath)) { throw 'Run Stage Check first.' }
    if ((Get-Content -LiteralPath $checkPath -Raw | ConvertFrom-Json).status -ne 'complete') {
        throw 'Stage Check must pass before any formal training.'
    }
}
if (Test-Path -LiteralPath $receiptPath) {
    $old = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($old.status -eq 'complete' -and $Stage -ne 'Check') {
        throw 'This stage is already complete. Do not overwrite it; inspect its receipt first.'
    }
}
$prerequisites = @{ M2Evaluate='M2Train'; M3Evaluate='M3Search' }
if ($prerequisites.ContainsKey($Stage)) {
    $prior = Join-Path $receiptRoot ($prerequisites[$Stage] + '.json')
    if (-not (Test-Path -LiteralPath $prior)) { throw 'Required training/search receipt missing.' }
    if ((Get-Content -LiteralPath $prior -Raw | ConvertFrom-Json).status -ne 'complete') {
        throw 'Required training/search did not complete.'
    }
}
if ($Stage -eq 'M2Train') {
    $models = Join-Path $RunRoot 'results\p7_corrected\m2\models'
    if ((Test-Path -LiteralPath $models) -and (Get-ChildItem -LiteralPath $models -File)) {
        throw 'Existing M2 models detected. Review/archive partial results; automatic overwrite is forbidden.'
    }
}
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
$log = Join-Path $logRoot ($Stage + '_' + $stamp + '.log')
$receipt = [ordered]@{stage=$Stage; status='running'; git_commit=$actualCommit;
    python=$Python; started_at=(Get-Date -Format o); output_root='results/p7_corrected';
    driver_sha256=(Get-FileHash -LiteralPath $PSCommandPath -Algorithm SHA256).Hash.ToLower();
    log=$log; config_hashes=@{} }
foreach ($config in @('methods/p7/m1_corrected.json','methods/p7/execution.m1-corrected.v2.json',
                      'methods/p7/m2_corrected.yaml','methods/p7/m3_corrected.json')) {
    $receipt.config_hashes[$config] = (Get-FileHash -LiteralPath $config -Algorithm SHA256).Hash.ToLower()
}
function Save-Receipt {
    $receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $receiptPath -Encoding UTF8
}
function Invoke-Python([string[]]$Arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previous
    if ($exitCode -ne 0) { throw ('Command failed with exit code ' + $exitCode) }
}
Save-Receipt
try {
    if ($Stage -eq 'Check') {
        Invoke-Python @('-c','import sys,torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); assert sys.version_info >= (3,11); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))')
        Invoke-Python @('methods/p7/run.py','preflight')
        Invoke-Python @('methods/p7/run_m1_full.py','plan')
    } elseif ($Stage -eq 'M1') {
        Invoke-Python @('methods/p7/run_m1_full.py','run','--confirm-start')
    } elseif ($Stage -eq 'M2Train') {
        Invoke-Python @('methods/m2/train.py','--config','methods/p7/m2_corrected.yaml')
    } elseif ($Stage -eq 'M2Evaluate') {
        Invoke-Python @('methods/m2/evaluate.py','--config','methods/p7/m2_corrected.yaml')
    } elseif ($Stage -eq 'M3Search') {
        Invoke-Python @('methods/m3/run.py','search','--config','methods/p7/m3_corrected.json','--output','results/p7_corrected/m3')
    } elseif ($Stage -eq 'M3Evaluate') {
        Invoke-Python @('methods/m3/run.py','evaluate','--config','methods/p7/m3_corrected.json','--output','results/p7_corrected/m3')
    }
    $receipt['status'] = 'complete'
} catch {
    $receipt['status'] = 'failed'
    $receipt['error'] = $_.Exception.Message
    throw
} finally {
    $receipt['completed_at'] = Get-Date -Format o
    Save-Receipt
    Copy-Item -LiteralPath $receiptPath -Destination (Join-Path $receiptRoot ($Stage + '_' + $stamp + '.json'))
}
Write-Output ('Stage complete: ' + $Stage)
Write-Output ('Receipt: ' + $receiptPath)
