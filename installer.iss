; Inno Setup script for BRAID
; AppVersion is supplied by scripts\build-release.ps1 from config.py.

#define AppName "BRAID"
#define AppExeName "BRAID.exe"
#ifndef AppVersion
  #error "AppVersion must be supplied with /DAppVersion=<version>"
#endif

[Setup]
AppId={{AA3B3513-E525-4CCC-AD39-3C2EEE733D3E}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=Tykocki Lab
AppPublisherURL=https://github.com/vr-oj/braid
AppSupportURL=https://github.com/vr-oj/braid/issues
AppUpdatesURL=https://github.com/vr-oj/braid/releases
DefaultDirName={autopf}\BRAID
DefaultGroupName=BRAID
OutputBaseFilename=BRAID_Setup_{#AppVersion}
OutputDir=installer_output
SetupIconFile=resources\app.ico
LicenseFile=LICENSE.md
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
PrivilegesRequired=lowest
SetupLogging=yes

[Files]
Source: "dist\BRAID\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\BRAID"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"
Name: "{autodesktop}\BRAID"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch BRAID"; Flags: nowait postinstall skipifsilent
