import sys
import os

# --- Configuration ---
app_version = os.environ.get('APP_VERSION', '0.9.5')
target_arch = os.environ.get('PYINSTALLER_TARGET_ARCH')
app_name = 'BRAID' if sys.platform == 'darwin' else f'BRAID_v{app_version}'

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
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name=app_name, # <-- DYNAMIC
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=True,
        icon=icon_file,
    )
