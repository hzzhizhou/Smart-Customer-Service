# 一键回归：检查依赖(MySQL/Redis) -> 必要时 docker compose 拉起 -> 全量 pytest
# 用法：在 RagServerSystem 目录下执行  .\run_tests.ps1
#       指定额外 pytest 参数：.\run_tests.ps1 -PytestArgs "-k intent"
param(
    [string]$PytestArgs = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
$venvPython = Join-Path $here "..\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { throw "未找到虚拟环境 Python：$venvPython" }

function Test-PortOpen([int]$Port, [string]$Name) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync("127.0.0.1", $Port)
        if ($task.Wait(1500) -and $client.Connected) { return $true }
        return $false
    } finally { $client.Close() }
}

# 1) 依赖预检：MySQL(3306) / Redis(6379)。测试全量 mock 外部服务，
#    但少数 DB 相关用例与本地环境相关，起不来时给出明确指引而不是让用例神秘失败。
$missing = @()
if (-not (Test-PortOpen 3306 "MySQL")) { $missing += "mysql" }
if (-not (Test-PortOpen 6379 "Redis")) { $missing += "redis" }

if ($missing.Count -gt 0) {
    Write-Host "依赖未就绪：$($missing -join ', ')，尝试 docker compose 拉起..." -ForegroundColor Yellow
    $composeFile = Join-Path $here "docker-compose.yml"
    if (Get-Command docker -ErrorAction SilentlyContinue) {
        docker compose -f $composeFile up -d 
        if ($LASTEXITCODE -ne 0) { throw "docker compose 启动失败，请手工检查 MySQL/Redis" }
        # 等待端口就绪（MySQL 首次初始化最长约 30s）
        $deadline = (Get-Date).AddSeconds(60)
        foreach ($svc in @(@("mysql",3306), @("redis",6379))) {
            $name, $port = $svc[0], $svc[1]
            while ((Get-Date) -lt $deadline -and -not (Test-PortOpen $port $name)) {
                Start-Sleep -Seconds 2
            }
            if (-not (Test-PortOpen $port $name)) {
                Write-Warning "$name 端口 $port 仍不可达；继续跑测试（大部分用例已 mock 外部依赖）"
            }
        }
    } else {
        Write-Warning "未检测到 docker，请手工启动 MySQL/Redis；继续跑测试（大部分用例已 mock 外部依赖）"
    }
}

# 2) 全量 pytest
Write-Host "`n=== 全量回归 pytest ===" -ForegroundColor Cyan
$sw = [System.Diagnostics.Stopwatch]::StartNew()
if ($PytestArgs -ne "") {
    & $venvPython -m pytest tests -q --tb=short $PytestArgs.Split(" ")
} else {
    & $venvPython -m pytest tests -q --tb=short
}
$code = $LASTEXITCODE
$sw.Stop()
Write-Host ("`n耗时 {0:N1}s，pytest 退出码 {1}" -f $sw.Elapsed.TotalSeconds, $code)
exit $code
