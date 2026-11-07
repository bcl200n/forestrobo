$ErrorActionPreference='Stop'
$statusPath='E:\forestrobo\ops\ssh-setup-status.json'
function Set-InstallStatus([string]$stage,[string]$message) {
    @{stage=$stage;message=$message;updated=(Get-Date).ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath $statusPath -Encoding UTF8
}
try {
    $principal=[Security.Principal.WindowsPrincipal]::new([Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {throw 'Administrator elevation required'}
    $package='E:\forestrobo\ops\downloads\OpenSSH-Win64-v10.4.0.0.msi'
    $expectedHash='788d8643464a01eb63640a2fd746f9ed23d106b5a901a35f88dc8cd40e446ac9'
    if ((Get-FileHash -LiteralPath $package -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedHash) {throw 'MSI SHA256 mismatch'}
    $signature=Get-AuthenticodeSignature -FilePath $package
    if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'CN=Microsoft Corporation,') {throw 'Microsoft MSI signature verification failed'}
    $previous=Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
    if ($previous.stage -eq 'complete') {exit 0}
    $worker=Get-CimInstance Win32_Process -Filter 'ProcessId=41764'
    if ($worker) {
        if ($previous.stage -ne 'installing' -or $worker.Name -ne 'powershell.exe' -or $worker.CommandLine -notlike '*E:\forestrobo\ops\setup-ssh.ps1*') {throw 'Original worker state changed; not interrupting it'}
        # Cancel only our stalled client. Never terminate Windows servicing processes.
        Stop-Process -Id $worker.ProcessId -ErrorAction Stop
        Set-InstallStatus 'cancelling_download' 'Waiting for original DISM host to close after cancelling our installer client'
        $deadline=(Get-Date).AddSeconds(60)
        do {
            $hostProcess=Get-CimInstance Win32_Process -Filter 'ProcessId=44508'
            if (-not $hostProcess -or $hostProcess.Name -ne 'DismHost.exe' -or $hostProcess.ParentProcessId -ne 41764) {break}
            Start-Sleep -Seconds 2
        } while ((Get-Date) -lt $deadline)
        if ($hostProcess -and $hostProcess.Name -eq 'DismHost.exe' -and $hostProcess.ParentProcessId -eq 41764) {throw 'Original DISM host is still cleaning up; MSI installation not started'}
    }
    Set-InstallStatus 'installing_msi' 'Installing Microsoft-signed Win32-OpenSSH 10.4.0.0 (official Preview release), server component only'
    $arguments='/i "'+$package+'" ADDLOCAL=Server /qn /norestart /L*v "E:\forestrobo\ops\ssh-msi-install.log"'
    $installer=Start-Process -FilePath "$env:WINDIR\System32\msiexec.exe" -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    if ($installer.ExitCode -notin @(0,3010)) {throw "MSI failed with code $($installer.ExitCode); see ssh-msi-install.log"}
    if ($installer.ExitCode -eq 3010) {Set-InstallStatus 'restart_required' 'MSI requires restart; not rebooted automatically';exit 2}
    & 'E:\forestrobo\ops\setup-ssh.ps1' -UseInstalledMsi
    exit $LASTEXITCODE
} catch {
    Set-InstallStatus 'failed' $_.Exception.Message
    $_ | Out-String | Add-Content -LiteralPath 'E:\forestrobo\ops\ssh-msi-wrapper.log'
    exit 1
}
