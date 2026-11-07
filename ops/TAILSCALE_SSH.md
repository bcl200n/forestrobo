# 本机通过 Tailscale 远程连接

本机 Windows 用户：`user`。Tailscale IPv4：`100.67.187.19`。
MagicDNS：`desktop-9g4b8ke.tail0b16b.ts.net`。

Windows 使用标准 OpenSSH Server，通过 Tailscale 网络连接。
运行安装脚本需要 Windows 管理员授权；Tailscale 登录成功并不代表 SSH 服务已安装。

## 安装和状态

如果没有看到从后台启动的 Windows 管理员提示，请双击桌面的
`完成Tailscale SSH配置.lnk`。这个入口从本机交互启动管理员授权，
并保留可见的 PowerShell 结果窗口。看到绿色 `SSH configuration COMPLETE.` 才表示配置成功。
桌面入口对应 `E:\forestrobo\ops\launch-ssh-setup.ps1`。

优先使用 Windows 可选功能；在管理员 PowerShell 中运行（安装器会保存配置备份）：

```powershell
& E:\forestrobo\ops\setup-ssh.ps1
```

本次 Windows 可选功能已安装成功，使用上面的 `setup-ssh.ps1` 配置即可。
以下备用 MSI 已下载但未安装，当前不需要运行。仅在 Windows 可选功能无法安装时使用：

```powershell
& E:\forestrobo\ops\install-ssh-msi.ps1
```

备用安装器只取消本次停滞安装的 PowerShell 客户端，并等待对应 DISM host 退出；
不强制终止系统更新或 Windows servicing 服务。若系统仍在清理，则停止并记录原因。
安装包为微软官方 Win32-OpenSSH `10.4.0.0p1-Preview` 的 Server 组件，
文件签名验证为 Microsoft Corporation，SHA-256 已与官方发布 API 提供的摘要比对：
`788d8643464a01eb63640a2fd746f9ed23d106b5a901a35f88dc8cd40e446ac9`。
该独立发行版需单独维护更新。

安装 MSI 版后，仅重新应用配置可运行：

```powershell
& E:\forestrobo\ops\setup-ssh.ps1 -UseInstalledMsi
```

检查安装结果：

```powershell
Get-Content E:\forestrobo\ops\ssh-setup-status.json
Get-Service sshd
```

只有状态为 `complete` 且 `sshd` 正在运行才表示本机安装完成。
安装日志位于 `E:\forestrobo\ops\ssh-setup.log`。
成功后的本机检查结果位于 `E:\forestrobo\ops\ssh-verification.json`，
包括服务、有效认证配置、防火墙过滤条件和服务器主机公钥指纹。

## 从远端连接

远端设备需要在线并加入可访问本机的 Tailscale 网络；tailnet 策略需允许访问本机 TCP 22。
在远端使用与已提供公钥配对的私钥（不要传输私钥到本机）：

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/id_ed25519 user@100.67.187.19
```

若私钥文件名不同，将 `~/.ssh/id_ed25519` 替换为其实际路径。
也可使用本机 MagicDNS 名称代替 IP。

Windows 远端可先检查端口：

```powershell
Test-NetConnection 100.67.187.19 -Port 22
```

首次连接应核对服务器主机公钥指纹。在本机运行：

```powershell
ssh-keygen -lf C:\ProgramData\ssh\ssh_host_ed25519_key.pub
```

授权的客户端公钥指纹为：
`SHA256:cEcKS7NQxNEpG0d26YsCbOudCKWuFMsTh2qe4V9ZhJw`。
它与服务器主机公钥指纹用途不同。

## 安装脚本配置范围

- SSH 用户限制为 `user`，仅允许公钥认证，关闭密码认证。
- 使用 `C:\Users\user\.ssh\authorized_keys`，不使用管理员组共享授权文件。
- Windows 防火墙 TCP 22 入站规则同时限制为 Tailscale 网卡、本机 `100.67.187.19` 地址、远端 `100.64.0.0/10` 地址段。
- 不配置路由器端口映射；不开放 RDP。
- `sshd` 设置为自动启动。休眠或关机期间无法连接。

如 Tailscale IP 发生变化，需用新的地址重新运行脚本：

```powershell
& E:\forestrobo\ops\setup-ssh.ps1 -TailscaleAddress '新的TailscaleIPv4'
```

本机端口与握手检查不能代替远端持有私钥的真实登录验证。
