import sys
import os
import re
import runpy
from pathlib import Path

# --- Configuration ---
config_scope = runpy.run_path('config.py')
app_version = os.environ.get('APP_VERSION', config_scope['APP_VERSION'])
target_arch = os.environ.get('PYINSTALLER_TARGET_ARCH')
app_name = 'BRAID'


def write_windows_version_resource():
    """Create Windows Explorer metadata from the release version."""
    if sys.platform != 'win32':
        return None

    match = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)(?:[-+].*)?', app_version)
    if not match:
        raise ValueError(f'APP_VERSION is not a supported release version: {app_version}')

    major, minor, patch = (int(part) for part in match.groups())
    numeric_version = f'({major}, {minor}, {patch}, 0)'
    version_path = Path('build') / 'BRAID-version-info.txt'
    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(
        f'''VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={numeric_version},
    prodvers={numeric_version},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable(
        '040904B0',
        [
          StringStruct('CompanyName', 'Tykocki Lab'),
          StringStruct('FileDescription', 'BRAID'),
          StringStruct('FileVersion', '{app_version}'),
          StringStruct('InternalName', 'BRAID'),
          StringStruct('LegalCopyright', 'Copyright (c) Tykocki Lab'),
          StringStruct('OriginalFilename', 'BRAID.exe'),
          StringStruct('ProductName', 'BRAID'),
          StringStruct('ProductVersion', '{app_version}')
        ]
      )
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)\n''',
        encoding='utf-8',
    )
    return str(version_path)


version_resource = write_windows_version_resource()

# --- Platform-specific icons ---
if sys.platform == 'darwin':  # macOS
    icon_file = os.path.join('resources', 'braid.icns')
else:  # Windows
    icon_file = os.path.join('resources', 'app.ico')

platform_datas = [(icon_file, 'icons')]

# --- PyInstaller Analysis ---
# This is where you define what gets included in your application.
a = Analysis(
    ['main.py'],  # <-- Your main script is the entry point
    pathex=[],
    binaries=[],
    datas=[
        ('resources', 'resources')  # <-- Add your resources folder here
    ],
    hiddenimports=[
        'scipy._cyutility'
    ],
    collect_stubs=['skimage'],
    hookspath=[],
    runtime_hooks=[],
    includes=['pyqtgraph.opengl'], # Not required for 2D plotting only. Remove if 3D plots are used.
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# --- Platform-specific Build Process ---
# All 'name' parameters now use the dynamic app_name variable.
if sys.platform == 'darwin':
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name, # <-- DYNAMIC
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=icon_file,
        target_arch=target_arch,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name, # <-- DYNAMIC
    )
    app = BUNDLE(
        coll,
        name=f"{app_name}.app", # <-- DYNAMIC
        icon=icon_file,
        bundle_identifier=None,
        version=app_version,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=app_name,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=icon_file,
        version=version_resource,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=app_name,
    )
