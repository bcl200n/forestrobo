$ErrorActionPreference='Stop'
$principal=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    try {
        # This is an interactive launcher: keep the elevated result window visible.
        $arguments='-NoProfile -ExecutionPolicy Bypass -NoExit -File "'+$PSCommandPath+'"'
        Start-Process -FilePath "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" -Verb RunAs -WindowStyle Normal -ArgumentList $arguments -ErrorAction Stop | Out-Null
    } catch {
        Write-Host 'Windows administrator authorization was not completed.' -ForegroundColor Red
        Write-Host $_.Exception.Message
        Read-Host 'Press Enter to close' | Out-Null
    }
    exit
}
$Host.UI.RawUI.WindowTitle='ForestRobo - Tailscale SSH setup'
Write-Host 'Applying Tailscale-only SSH configuration...' -ForegroundColor Cyan
& "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File 'E:\forestrobo\ops\setup-ssh.ps1'
$setupExitCode=$LASTEXITCODE
$state=Get-Content -LiteralPath 'E:\forestrobo\ops\ssh-setup-status.json' -Raw | ConvertFrom-Json
if ($setupExitCode -eq 0 -and $state.stage -eq 'complete') {
    Write-Host 'SSH configuration COMPLETE.' -ForegroundColor Green
    Write-Host 'Remote connection (use the private key matching your supplied public key):'
    Write-Host 'ssh -i ~/.ssh/id_ed25519 user@100.67.187.19' -ForegroundColor Green
    Get-Content -LiteralPath 'E:\forestrobo\ops\ssh-verification.json' -Raw | Write-Host
} else {
    Write-Host ('Configuration is not complete: '+$state.message) -ForegroundColor Red
    Write-Host 'Log: E:\forestrobo\ops\ssh-setup.log'
    Get-Content -LiteralPath 'E:\forestrobo\ops\ssh-setup.log' -Tail 20 | Write-Host
}
Write-Host 'You may close this window after reading the result.'
