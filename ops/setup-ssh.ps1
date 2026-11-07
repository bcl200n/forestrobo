param(
    [string]$RemoteAddress='100.64.0.0/10',
    [string]$TailscaleAddress='100.67.187.19',
    [switch]$UseInstalledMsi
)
$ErrorActionPreference='Stop'
$resultPath='E:\forestrobo\ops\ssh-setup-status.json'
$logPath='E:\forestrobo\ops\ssh-setup.log'
function Status([string]$stage,[string]$message='') {
    @{stage=$stage;message=$message;updated=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $resultPath -Encoding UTF8
}
try {
    Start-Transcript -Path $logPath -Append | Out-Null
    $principal=[Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {throw 'Administrator elevation required'}
    $adapter=Get-NetIPAddress -AddressFamily IPv4 -IPAddress $TailscaleAddress -ErrorAction Stop
    if ($adapter.InterfaceAlias -ne 'Tailscale') {throw 'Expected address is not on the Tailscale adapter'}
    if (-not $UseInstalledMsi) {
        Status 'installing' 'Installing Windows OpenSSH Server optional capability'
        $cap=Get-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
        if ($cap.State -ne 'Installed') {
            $installation=Add-WindowsCapability -Online -Name 'OpenSSH.Server~~~~0.0.1.0'
            if ($installation.RestartNeeded) {Status 'restart_required' 'Windows requires a restart; not rebooted automatically';exit 2}
        }
    }
    if ($UseInstalledMsi) {
        $binary=@('C:\Program Files\OpenSSH\sshd.exe','C:\Program Files\OpenSSH-Win64\sshd.exe') | Where-Object {Test-Path -LiteralPath $_} | Select-Object -First 1
    } else {$binary=Join-Path $env:WINDIR 'System32\OpenSSH\sshd.exe'}
    if (-not $binary) {throw 'Installed OpenSSH Server binary not found'}
    if (-not (Test-Path -LiteralPath $binary)) {throw 'Server binary unavailable after installation'}
    # Limit the optional feature's auto-created rule before starting the server.
    $rule=Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -ErrorAction SilentlyContinue
    if ($rule) {Set-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -Enabled True -Direction Inbound -Action Allow -Profile Any -RemoteAddress $RemoteAddress -LocalAddress $TailscaleAddress -InterfaceAlias 'Tailscale'}
    else {New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server TCP 22 (Tailscale, key only)' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 22 -Profile Any -RemoteAddress $RemoteAddress -LocalAddress $TailscaleAddress -InterfaceAlias 'Tailscale' | Out-Null}
    Set-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -NewDisplayName 'OpenSSH Server TCP 22 (Tailscale, key only)'
    Status 'configuring' 'Configuring per-user public key authentication'
    $userAccount=Get-LocalUser -Name 'user'
    $sid=$userAccount.SID.Value
    $keyDirectory='C:\Users\user\.ssh'
    $keyPath=Join-Path $keyDirectory 'authorized_keys'
    if (-not (Test-Path -LiteralPath $keyPath)) {throw 'Staged authorized_keys is missing'}
    $expected='AAAAC3NzaC1lZDI1NTE5AAAAIJSc5dFu6U3RmevMX6hRayGSOfZaQQtUWP631YWZg4J6'
    if (-not ((Get-Content -LiteralPath $keyPath) -match [regex]::Escape($expected))) {throw 'Expected authorized public key is missing'}
    # Explicit ACL: owner, SYSTEM, Administrators only, language-independent SIDs.
    foreach ($path in @($keyDirectory,$keyPath)) {
        $acl=Get-Acl -LiteralPath $path
        $acl.SetAccessRuleProtection($true,$false)
        foreach ($entry in @($acl.Access)) {$acl.RemoveAccessRuleSpecific($entry)}
        $acl.SetOwner([Security.Principal.SecurityIdentifier]::new($sid))
        foreach ($allowedSid in @($sid,'S-1-5-18','S-1-5-32-544')) {
            $access=[Security.AccessControl.FileSystemAccessRule]::new(
                [Security.Principal.SecurityIdentifier]::new($allowedSid),
                [Security.AccessControl.FileSystemRights]::FullControl,
                [Security.AccessControl.AccessControlType]::Allow)
            $acl.AddAccessRule($access)
        }
        Set-Acl -LiteralPath $path -AclObject $acl
    }
    $sshRoot=Join-Path $env:ProgramData 'ssh'
    New-Item -ItemType Directory -Force -Path $sshRoot | Out-Null
    $configPath=Join-Path $sshRoot 'sshd_config'
    $backupPath=$configPath+'.backup-'+(Get-Date -Format yyyyMMddHHmmss)
    if (Test-Path -LiteralPath $configPath) {Copy-Item -LiteralPath $configPath -Destination $backupPath}
    # No shared administrators_authorized_keys: scope this key to this account.
    $config=@'
Port 22
AddressFamily inet
ListenAddress 0.0.0.0
PubkeyAuthentication yes
AuthenticationMethods publickey
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
AuthorizedKeysFile .ssh/authorized_keys
StrictModes yes
AllowUsers user
AllowAgentForwarding no
AllowTcpForwarding yes
GatewayPorts no
MaxAuthTries 3
LoginGraceTime 30
LogLevel VERBOSE
Subsystem sftp sftp-server.exe
'@
    [IO.File]::WriteAllText($configPath,$config,[Text.UTF8Encoding]::new($false))
    & (Join-Path (Split-Path $binary) 'ssh-keygen.exe') -A
    if ($LASTEXITCODE -ne 0) {throw 'Host key generation failed'}
    & $binary -t -f $configPath
    if ($LASTEXITCODE -ne 0) {
        if (Test-Path -LiteralPath $backupPath) {Copy-Item -LiteralPath $backupPath -Destination $configPath -Force}
        throw 'sshd configuration validation failed; no service restart performed'
    }
    New-Item -Path 'HKLM:\SOFTWARE\OpenSSH' -Force | Out-Null
    New-ItemProperty -Path 'HKLM:\SOFTWARE\OpenSSH' -Name DefaultShell -PropertyType String -Value "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe" -Force | Out-Null
    Set-Service -Name sshd -StartupType Automatic
    if ((Get-Service sshd).Status -eq 'Running') {Restart-Service sshd} else {Start-Service sshd}
    Start-Sleep -Seconds 2
    $listener=Get-NetTCPConnection -LocalPort 22 -State Listen -ErrorAction Stop
    $firewall=Get-NetFirewallRule -Name 'OpenSSH-Server-In-TCP'
    $addressFilter=$firewall | Get-NetFirewallAddressFilter
    $interfaceFilter=$firewall | Get-NetFirewallInterfaceFilter
    $portFilter=$firewall | Get-NetFirewallPortFilter
    [ordered]@{
        binary=$binary
        service=Get-CimInstance Win32_Service -Filter "Name='sshd'" | Select-Object Name,State,StartMode,PathName
        firewall=[ordered]@{enabled=[string]$firewall.Enabled;action=[string]$firewall.Action;localAddress=$addressFilter.LocalAddress;remoteAddress=$addressFilter.RemoteAddress;interfaceAlias=$interfaceFilter.InterfaceAlias;localPort=$portFilter.LocalPort;protocol=$portFilter.Protocol}
        hostFingerprint=(& (Join-Path (Split-Path $binary) 'ssh-keygen.exe') -lf (Join-Path $sshRoot 'ssh_host_ed25519_key.pub') | Out-String).Trim()
        effectiveConfig=@(& $binary -T -f $configPath | Where-Object {$_ -match '^(pubkeyauthentication|passwordauthentication|authenticationmethods|authorizedkeysfile|allowusers|kbdinteractiveauthentication) '})
        checked=(Get-Date).ToString('o')
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath 'E:\forestrobo\ops\ssh-verification.json' -Encoding UTF8
    Status 'complete' "sshd running; TCP 22 listening; Tailscale interface only; local address: $TailscaleAddress; remote scope: $RemoteAddress; key-only login for user"
} catch {
    Status 'failed' $_.Exception.Message
    Write-Error $_ -ErrorAction Continue
    exit 1
} finally {
    Stop-Transcript -ErrorAction SilentlyContinue | Out-Null
}
