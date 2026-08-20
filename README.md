<p align="center">
  <img src="https://github.com/vr-oj/braid/blob/main/resources/icon_128.png?raw=true" alt="BRAID Logo">
</p>

<h3 align="center">BRAID</h3>

<p align="center">
  Data processing app custom-made for BUTI data.
  <br>
  <br>
  <a href="https://pubmed.ncbi.nlm.nih.gov/41690611/">BUTI Device</a>
  -
  <a href="http://www.tykocki-lab.com/">Tykocki Lab</a>
</p>


## BRAID
BUTI Research Analysis & Inspection Dashboard

A cross-platform application for performing mechanical analyses of BUTI experimental data.

## Opening a recording from another application

BRAID accepts a generic `--open <recording-path>` launch argument for TIFF,
AVI, and MKV files. This lets acquisition tools hand BRAID an already-saved
recording path without sharing application code or creating a runtime
dependency; BRAID continues to work normally as a standalone analysis app.

## macOS installation

Download the DMG that matches the Mac: `Silicon` for Apple silicon (M-series)
or `Intel` for an Intel processor. Open the DMG and copy `BRAID.app` to the
Applications folder.

BRAID is not currently signed with an Apple Developer ID. On first launch,
macOS may block it as an app from an unidentified developer. Try to open BRAID
once, then open **System Settings > Privacy & Security**, scroll to **Security**,
and choose **Open Anyway**. Apple documents this process in
[Open a Mac app from an unknown developer](https://support.apple.com/guide/mac-help/open-a-mac-app-from-an-unknown-developer-mh40616/mac).

Only override this warning for a BRAID DMG downloaded from the project's
official GitHub release. A message that the app is damaged is not the expected
unsigned-app warning; re-download the matching architecture build and report
the problem rather than bypassing it with a Terminal command.
