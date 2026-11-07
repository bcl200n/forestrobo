$ErrorActionPreference='Stop'
$url='http://127.0.0.1:8766/rescue.html'
try {
    $existing=Invoke-RestMethod -Uri 'http://127.0.0.1:8766/api/rescue/catalog' -TimeoutSec 2
    if ($existing.catalog.schema -eq 'forestrobo-modular-v1') {
        Start-Process -FilePath $url -WindowStyle Normal
        exit
    }
} catch {
    # No existing demo; start the interactive local service below.
}
Write-Host 'Starting local rescue demo. Close this window to stop the demo API.'
Write-Host $url -ForegroundColor Green
& 'E:\forestrobo\rl\.venv\Scripts\python.exe' -u 'E:\forestrobo\scripts\serve_rescue.py' --port 8766 --open-browser
