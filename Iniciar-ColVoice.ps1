param([ValidateSet('cuda', 'cpu')][string]$Device = 'cuda')
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
$taskUrl = 'http://127.0.0.1:8001'
if (-not (Test-Path -LiteralPath $taskPython)) { throw 'Falta el entorno Python de ColVoice.' }
try {
    $response = Invoke-WebRequest -UseBasicParsing "$taskUrl/config" -TimeoutSec 3
    if ($response.StatusCode -eq 200 -and $response.Content -match 'ColVoice|OmniVoice') {
        Start-Process $taskUrl
        exit 0
    }
} catch { }
New-Item -ItemType Directory -Force (Join-Path $taskRoot 'logs') | Out-Null
$taskProcess = Start-Process -FilePath $taskPython -ArgumentList "-u run_omnivoice.py --device $Device" -WorkingDirectory $taskRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $taskRoot 'logs\server.log') -RedirectStandardError (Join-Path $taskRoot 'logs\server.err') -PassThru
$taskProcess.Id | Set-Content (Join-Path $taskRoot 'logs\server.pid')
Write-Host 'Cargando ColVoice. El navegador se abrira cuando este listo...'
for ($attempt = 0; $attempt -lt 150; $attempt++) {
    Start-Sleep -Seconds 2
    $taskProcess.Refresh()
    if ($taskProcess.HasExited) {
        Get-Content (Join-Path $taskRoot 'logs\server.err') -Tail 25
        throw 'ColVoice no pudo iniciar. Revisa logs\server.err.'
    }
    try {
        $response = Invoke-WebRequest -UseBasicParsing "$taskUrl/config" -TimeoutSec 2
        if ($response.StatusCode -eq 200 -and $response.Content -match 'ColVoice|OmniVoice') {
            Start-Process $taskUrl
            exit 0
        }
    } catch { }
}
Write-Host "La carga continua. Abre $taskUrl cuando logs\server.log indique que esta listo."
