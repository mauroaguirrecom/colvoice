$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot 'logs\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host 'No hay una instancia registrada.'; exit 0 }
$serverProcessId = [int](Get-Content -LiteralPath $pidFile)
$serverProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$serverProcessId"
$expectedPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not $serverProcess) { Write-Host 'MAGI Eleven ya esta detenido.'; exit 0 }
if (-not $serverProcess.CommandLine.Contains($expectedPython) -or -not $serverProcess.CommandLine.Contains('run_omnivoice.py')) {
    throw 'El proceso registrado no corresponde a MAGI Eleven; no se ha detenido.'
}
& taskkill.exe /PID $serverProcessId /T /F
if ($LASTEXITCODE -ne 0) { throw 'No se pudo detener MAGI Eleven.' }
