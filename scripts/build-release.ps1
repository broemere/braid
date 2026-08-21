[CmdletBinding()]
param(
    [string]$PythonPath = ".venv\Scripts\python.exe",
    [string]$InnoCompiler = "",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath $PythonPath) {
        $python = (Resolve-Path -LiteralPath $PythonPath).Path
    }
    else {
        $pythonCommand = Get-Command $PythonPath -CommandType Application -ErrorAction Stop
        $python = $pythonCommand.Source
    }
    $config = Get-Content -LiteralPath "config.py" -Raw
    if ($config -notmatch '(?m)^APP_VERSION\s*=\s*["'']([^"'']+)["'']') {
        throw "APP_VERSION was not found in config.py."
    }
    $version = $Matches[1]
    if ($version -notmatch '^\d+\.\d+\.\d+(-(alpha|beta|rc)\.\d+)?$') {
        throw "Invalid APP_VERSION in config.py: $version"
    }

    & $python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "The Python environment contains broken requirements."
    }

    if (-not $SkipTests) {
        $oldQtPlatform = $env:QT_QPA_PLATFORM
        $env:QT_QPA_PLATFORM = "offscreen"
        try {
            & $python -m unittest discover -s tests -v
            if ($LASTEXITCODE -ne 0) {
                throw "The BRAID test suite failed."
            }
        }
        finally {
            $env:QT_QPA_PLATFORM = $oldQtPlatform
        }
    }

    $oldAppVersion = $env:APP_VERSION
    $env:APP_VERSION = $version
    try {
        & $python -m PyInstaller --noconfirm --clean build.spec
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed."
        }
    }
    finally {
        $env:APP_VERSION = $oldAppVersion
    }

    $application = Join-Path $repoRoot "dist\BRAID\BRAID.exe"
    if (-not (Test-Path -LiteralPath $application)) {
        throw "Expected application was not created: $application"
    }

    if ($InnoCompiler) {
        $iscc = (Resolve-Path -LiteralPath $InnoCompiler).Path
    }
    else {
        $innoCandidates = @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )
        $iscc = $innoCandidates |
            Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
            Select-Object -First 1
    }

    if (-not $iscc) {
        throw "Inno Setup 6 was not found. Install it or pass -InnoCompiler."
    }

    & $iscc "/DAppVersion=$version" "installer.iss"
    if ($LASTEXITCODE -ne 0) {
        throw "Inno Setup failed."
    }

    $installer = Join-Path $repoRoot "installer_output\BRAID_Setup_$version.exe"
    if (-not (Test-Path -LiteralPath $installer)) {
        throw "Expected installer was not created: $installer"
    }

    $hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $installer).Hash.ToLowerInvariant()
    $filename = Split-Path -Leaf $installer
    $checksum = "$installer.sha256"
    [System.IO.File]::WriteAllText(
        $checksum,
        "$hash  $filename`r`n",
        [System.Text.Encoding]::ASCII
    )

    Write-Host ""
    Write-Host "BRAID v$version release installer is ready:" -ForegroundColor Green
    Write-Host "  Installer: $installer"
    Write-Host "  Checksum:  $checksum"
    Write-Host "  SHA-256:   $hash"
}
finally {
    Pop-Location
}
