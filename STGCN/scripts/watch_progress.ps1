# 即時顯示 RWF-2000 骨架抽取與訓練流程進度（每 5 秒更新，Ctrl+C 結束；只讀取不影響背景程序）
# 用法：powershell -NoExit -ExecutionPolicy Bypass -File STGCN\scripts\watch_progress.ps1
$ErrorActionPreference = "SilentlyContinue"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$root = Split-Path -Parent $PSScriptRoot
$sessions = Join-Path $root "data\interim\sessions"
$raw = Join-Path $root "data\raw\rwf2000"
$pipeLog = Join-Path $root "logs\rwf2000_pipeline.log"
$host.UI.RawUI.WindowTitle = "RWF-2000 訓練進度"

$total = (Get-ChildItem $raw -Recurse -File -Include *.avi, *.mp4, *.mkv, *.mov).Count
$history = New-Object System.Collections.Generic.List[object]

while ($true) {
    $dirs = Get-ChildItem $sessions -Directory -Filter "rwf2000_*"
    $done = $dirs.Count
    $now = Get-Date
    $history.Add([pscustomobject]@{ t = $now; n = $done })
    while ($history.Count -gt 0 -and ($now - $history[0].t).TotalSeconds -gt 300) { $history.RemoveAt(0) }

    $rate = 0
    if ($history.Count -ge 2) {
        $dt = ($history[$history.Count - 1].t - $history[0].t).TotalMinutes
        if ($dt -gt 0) { $rate = ($history[$history.Count - 1].n - $history[0].n) / $dt }
    }

    Clear-Host
    Write-Host "RWF-2000 骨架抽取 → 前處理 → 訓練 → 評估        $($now.ToString('HH:mm:ss'))" -ForegroundColor Cyan
    Write-Host ("=" * 64)
    $pct = if ($total -gt 0) { [math]::Floor(100 * $done / $total) } else { 0 }
    $bar = ("#" * [math]::Floor($pct / 2)).PadRight(50, ".")
    Write-Host ("骨架抽取 [{0}] {1,3}%  {2}/{3}" -f $bar, $pct, $done, $total)
    if ($rate -gt 0 -and $done -lt $total) {
        $eta = $now.AddMinutes(($total - $done) / $rate)
        Write-Host ("速度 {0:N0} 支/分鐘，預計 {1} 抽取完成" -f $rate, $eta.ToString("HH:mm"))
    }
    Write-Host ""
    Write-Host "各類別已完成：" -ForegroundColor Yellow
    $dirs | ForEach-Object { ($_.Name -split "_")[1..3] -join "/" } | Group-Object | Sort-Object Name |
        ForEach-Object { Write-Host ("  {0,-28} {1,5}" -f $_.Name, $_.Count) }

    Write-Host ""
    Write-Host "最新處理：" -ForegroundColor Yellow
    $dirs | Sort-Object LastWriteTime -Descending | Select-Object -First 3 |
        ForEach-Object { Write-Host ("  {0}  {1}" -f $_.LastWriteTime.ToString("HH:mm:ss"), $_.Name) }

    Write-Host ""
    Write-Host "訓練流程紀錄（logs\rwf2000_pipeline.log 最後 12 行）：" -ForegroundColor Yellow
    if (Test-Path $pipeLog) {
        Get-Content $pipeLog -Tail 12 -Encoding UTF8 | ForEach-Object { Write-Host "  $_" }
    }
    Write-Host ""
    Write-Host "（每 5 秒更新；關閉此視窗不會影響背景程序）" -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}
