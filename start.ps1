# 一鍵啟動：各開一個視窗執行後端（FastAPI :8000）與前端（Vite :5173），並開啟瀏覽器
# 用法（在 attack-prediction 資料夾）：
#   powershell -ExecutionPolicy Bypass -File start.ps1
# 關閉：直接關掉「後端」「前端」兩個視窗
$root = $PSScriptRoot

function Test-Port($port) {
    return [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
}

if (Test-Port 8000) {
    Write-Host "後端已在執行（port 8000），略過"
} else {
    $be = "`$host.UI.RawUI.WindowTitle='後端 FastAPI :8000'; `$env:PYTHONIOENCODING='utf-8'; " +
          "Set-Location '$root\backend'; & .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000"
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $be
    Write-Host "已啟動後端 → http://127.0.0.1:8000/docs"
}

if (Test-Port 5173) {
    Write-Host "前端已在執行（port 5173），略過"
} else {
    $fe = "`$host.UI.RawUI.WindowTitle='前端 Vite :5173'; Set-Location '$root\frontend\attack-frontend'; " +
          "pnpm dev --host 127.0.0.1 --port 5173"
    Start-Process powershell -ArgumentList '-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $fe
    Write-Host "已啟動前端 → http://127.0.0.1:5173"
}

# 等前端就緒後開啟瀏覽器
for ($i = 0; $i -lt 40; $i++) {
    if (Test-Port 5173) { break }
    Start-Sleep -Seconds 1
}
Start-Process "http://127.0.0.1:5173"
