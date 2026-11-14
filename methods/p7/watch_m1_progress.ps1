param(
    [string]$RunRoot = 'C:\Users\Administrator\Desktop\Security\p7-run-3632afa',
    [ValidateRange(2,60)][int]$IntervalSeconds = 5,
    [switch]$Once
)
$ErrorActionPreference = 'Stop'
$receiptPath = Join-Path $RunRoot 'results\p7_corrected\receipts\M1.json'
$formalRoot = Join-Path $RunRoot 'results\p7_corrected\m1\formal'
while ($true) {
    $receipt = $null
    if (Test-Path -LiteralPath $receiptPath) {
        try { $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json }
        catch { Start-Sleep -Seconds 2; continue }
    }
    Clear-Host
    Write-Host 'P7 M1 corrected - read-only progress monitor' -ForegroundColor Cyan
    Write-Host ('Updated: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
    Write-Host 'This window never starts or stops training. Ctrl+C stops only this monitor.'
    Write-Host ''
    if ($null -eq $receipt) {
        Write-Host 'Waiting for the M1 stage receipt...'
        Start-Sleep -Seconds $IntervalSeconds
        continue
    }
    $started = [DateTimeOffset]::Parse($receipt.started_at)
    $elapsed = [DateTimeOffset]::Now - $started
    Write-Host ('Stage status: ' + $receipt.status)
    Write-Host ('Elapsed: ' + $elapsed.ToString('d\.hh\:mm\:ss'))
    Write-Host ('Git commit: ' + $receipt.git_commit)
    $run = $null
    if (Test-Path -LiteralPath $formalRoot) {
        $candidates = foreach ($folder in (Get-ChildItem -LiteralPath $formalRoot -Directory)) {
            $identityPath = Join-Path $folder.FullName 'run.json'
            if (Test-Path -LiteralPath $identityPath) {
                try {
                    $identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
                    $created = [DateTimeOffset]::Parse($identity.created_at)
                    if ($created -ge $started.AddSeconds(-2)) {
                        [pscustomobject]@{ Folder=$folder; Created=$created }
                    }
                } catch { }
            }
        }
        $latest = $candidates | Sort-Object Created -Descending | Select-Object -First 1
        if ($null -ne $latest) { $run = $latest.Folder }
    }
    if ($null -ne $run) {
        $jobRoot = Join-Path $run.FullName 'jobs'
        $recordRoot = Join-Path $run.FullName 'records'
        $jobs = @(Get-ChildItem -LiteralPath $jobRoot -Filter '*.json' -File -ErrorAction SilentlyContinue)
        $records = @(Get-ChildItem -LiteralPath $recordRoot -Filter '*.json' -File -ErrorAction SilentlyContinue)
        Write-Host ('Run: ' + $run.Name)
        Write-Host ('Completed training + validation checkpoints: ' + $jobs.Count + ' / 10')
        Write-Host ('Saved Test threshold records: ' + $records.Count + ' / 20')
        Write-Progress -Id 1 -Activity 'Training and validation checkpoints' -Status ($jobs.Count.ToString() + '/10 complete') -PercentComplete ([Math]::Min(100,10*$jobs.Count))
        if (Test-Path -LiteralPath (Join-Path $run.FullName 'validation_frozen.json')) {
            Write-Host 'All validation models and thresholds are frozen. Test preparation/evaluation is allowed.'
            Write-Progress -Id 2 -Activity 'Test threshold records' -Status ($records.Count.ToString() + '/20 saved') -PercentComplete ([Math]::Min(100,5*$records.Count))
        } else {
            Write-Host 'Phase: data preparation, fitting or validation. Test gate remains closed.'
        }
        if ($jobs.Count -gt 0) {
            Write-Host ('Latest completed task: ' + ($jobs | Sort-Object LastWriteTime -Descending | Select-Object -First 1).BaseName)
        }
        $failedPath = Join-Path $run.FullName 'FAILED.json'
        if (Test-Path -LiteralPath $failedPath) {
            Write-Host 'Run failed. Keep the training log and FAILED.json for diagnosis.' -ForegroundColor Red
            Get-Content -LiteralPath $failedPath
            break
        }
    } else {
        Write-Host 'Waiting for the formal run directory...'
    }
    Write-Host ''
    Write-Host 'Latest training log:' -ForegroundColor Yellow
    if ($receipt.log -and (Test-Path -LiteralPath $receipt.log)) {
        Get-Content -LiteralPath $receipt.log -Tail 8 | ForEach-Object { Write-Host $_ }
    }
    Write-Host ''
    Write-Host 'Progress counts completed tasks; it does not estimate progress within one tree fit.'
    if ($Once -or $receipt.status -in @('complete','failed')) { break }
    Start-Sleep -Seconds $IntervalSeconds
}
Write-Progress -Id 1 -Activity 'Training and validation checkpoints' -Completed
Write-Progress -Id 2 -Activity 'Test threshold records' -Completed
