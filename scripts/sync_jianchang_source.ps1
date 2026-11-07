param(
    [ValidatePattern('^[a-zA-Z0-9_.:-]+$')][string]$RemoteHost = '100.109.188.62',
    [ValidatePattern('^[a-zA-Z0-9_.-]+$')][string]$RemoteUser = 'chunlin',
    [string]$Destination = (Join-Path $PSScriptRoot '../data/jianchang/source')
)

$ErrorActionPreference = 'Stop'
$OutputEncoding = [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$destinationRoot = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Path $destinationRoot -Force | Out-Null
$sshOptions = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=15',
                '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3')

# Only the CAD versions and map image identified by the user are included.
$remoteScript = @'
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$cadRoot = 'E:\大论文\大论文写作\建昌古城cad数据'
$planRoot = 'E:\大论文\大论文写作\西昌建昌古城平面图'
$files = @()
foreach ($file in Get-ChildItem -LiteralPath $cadRoot -File -Filter '*.dwg') {
    $files += [PSCustomObject]@{ Source=$file.FullName; Group='cad'; Name=$file.Name;
        Bytes=$file.Length; ModifiedUtc=$file.LastWriteTimeUtc.ToString('o');
        Sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
}
foreach ($name in @('建昌古城cad测绘数据3_2010dwg.dwg', '5567cc016f7b9cda50c8b9bf00d5418a.png')) {
    $file = Get-Item -LiteralPath (Join-Path $planRoot $name)
    $files += [PSCustomObject]@{ Source=$file.FullName; Group='plan'; Name=$file.Name;
        Bytes=$file.Length; ModifiedUtc=$file.LastWriteTimeUtc.ToString('o');
        Sha256=(Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash }
}
$nearby = @(Get-ChildItem -LiteralPath 'E:\大论文\大论文写作' -Directory |
    Where-Object { $_.Name -match '无人机|航拍|倾斜|三维|影像|实景|测绘|DOM|DSM|古城' } |
    Select-Object -ExpandProperty FullName)
[PSCustomObject]@{ Files=$files; NearbyDataDirectories=$nearby } | ConvertTo-Json -Depth 4 -Compress
'@
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($remoteScript))
$response = & ssh @sshOptions -l $RemoteUser $RemoteHost "powershell.exe -NoProfile -NonInteractive -EncodedCommand $encoded"
if ($LASTEXITCODE -ne 0) { throw 'Could not read the remote source manifest.' }
$inventory = ($response -join "`n") | ConvertFrom-Json
$manifest = [ordered]@{ RemoteHost=$RemoteHost; RemoteUser=$RemoteUser;
    Status='copying'; Files=$inventory.Files; NearbyDataDirectories=$inventory.NearbyDataDirectories }
$manifestPath = Join-Path $destinationRoot 'manifest.json'
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8

foreach ($file in $inventory.Files) {
    if ($file.Group -notin @('cad', 'plan') -or [IO.Path]::GetFileName($file.Name) -ne $file.Name) {
        throw 'Unexpected destination path in source manifest.'
    }
    $folder = Join-Path $destinationRoot $file.Group
    New-Item -ItemType Directory -Path $folder -Force | Out-Null
    $target = [IO.Path]::GetFullPath((Join-Path $folder $file.Name))
    if (-not $target.StartsWith($destinationRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Destination must stay inside the selected data directory.'
    }
    if ((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $file.Sha256) {
        Write-Output ('Verified existing ' + $file.Name)
        continue
    }
    $partial = $target + '.partial'
    Write-Output ('Copying ' + $file.Name + ' (' + $file.Bytes + ' bytes)')
    $quotedSource = $file.Source.Replace("'", "''")
    $readScript = @"
`$ErrorActionPreference = 'Stop'
`$ProgressPreference = 'SilentlyContinue'
`$sourceStream = [IO.File]::OpenRead('$quotedSource')
try {
    `$sourceStream.CopyTo([Console]::OpenStandardOutput())
} finally {
    `$sourceStream.Dispose()
}
"@
    $readCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($readScript))
    $processInfo = [Diagnostics.ProcessStartInfo]::new()
    $processInfo.FileName = (Get-Command ssh).Source
    $processInfo.Arguments = ($sshOptions -join ' ') + " -l $RemoteUser $RemoteHost " +
        '"powershell.exe -NoProfile -NonInteractive -EncodedCommand ' + $readCommand + '"'
    $processInfo.UseShellExecute = $false
    $processInfo.CreateNoWindow = $true
    $processInfo.RedirectStandardOutput = $true
    $processInfo.RedirectStandardError = $true
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $processInfo
    $outputStream = [IO.File]::Create($partial)
    try {
        [void]$process.Start()
        $errors = $process.StandardError.ReadToEndAsync()
        # Preserve binary bytes directly. PowerShell's text pipeline would corrupt DWG/PNG data.
        $process.StandardOutput.BaseStream.CopyTo($outputStream)
        $process.WaitForExit()
        $transferError = $errors.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) { throw ('SSH transfer failed: ' + $transferError) }
    } finally {
        $outputStream.Dispose()
        $process.Dispose()
    }
    if ((Get-Item -LiteralPath $partial).Length -ne $file.Bytes -or
        (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $file.Sha256) {
        throw ('Source verification failed for ' + $file.Source)
    }
    Move-Item -LiteralPath $partial -Destination $target -Force
    Write-Output ('Verified copy ' + $file.Name)
}
$manifest.Status = 'verified'
$manifest['VerifiedUtc'] = [DateTime]::UtcNow.ToString('o')
$manifest | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifestPath -Encoding UTF8
Write-Output ('Complete. Manifest saved to ' + $manifestPath)
Write-Output ('Nearby data directories: ' + ($inventory.NearbyDataDirectories -join ', '))
