# Isolated Store path diagnostic

Version **1.1.0.0** tests actual pip installation using resolved paths.
This test package has identity `UWCC.StorePathProbe`, distinct from UWCC. Keep
the existing UWCC installation. The probe only creates its own fresh runtime
and marker script under `%LOCALAPPDATA%\UWCC.StorePathProbe`, subject to MSIX
redirection. It downloads `get-pip.py` from bootstrap.pypa.io and installs pip
only inside that fresh diagnostic runtime. Internet access is required. It does
not install PyTorch or models, or access UWCC settings or runtimes. Each launch
uses a new directory.

## Run on the affected PC

1. Copy `dist\store-path-probe\UWCC.StorePathProbe-test.zip` to that PC and
   extract it. It contains the MSIX, its public test certificate and these
   instructions. The private signing key is not included.
2. An administrator must trust this dedicated test certificate. Open PowerShell
   **as administrator**, change to the extracted folder, and run:

   ```powershell
   Import-Certificate -FilePath .\UWCC.StorePathProbe.cer -CertStoreLocation Cert:\LocalMachine\TrustedPeople
   ```

   This grants trust to packages signed with this test certificate. The
   certificate expires after 30 days and is for this diagnostic only.
3. Close the administrator shell. Under the **affected standard-user account**,
   open ordinary PowerShell in the extracted folder and run:

   ```powershell
   Add-AppxPackage -Path .\UWCC.StorePathProbe.msix
   ```

4. Launch **UWCC Store Path Probe (diagnostic)** from Start. Do not launch the
   unpackaged executable; package identity is essential to this experiment.
   Close the previous probe before installing the update. The same certificate
   is reused, so if it is already trusted, skip step 2. Installing 1.1.0.0
   updates the existing diagnostic package; no uninstall is needed.
   Allow a few minutes for download and pip installation.
5. After the completion dialog, the JSON report opens in Notepad. Save a copy
   somewhere convenient and send its contents back. It includes local paths
   and therefore the Windows user name; redact the user name if desired.

If installation fails, send the complete PowerShell error instead. Do not
uninstall UWCC or change its folders to make the probe install.

## Interpretation

The report records package identity, logical versus resolved physical paths,
and a marker-script run using physical paths. It then verifies that pip is
initially absent, enables `import site`, downloads and runs `get-pip.py`, and
checks `python -I -m pip --version`. Look for `probe_version: "1.1.0.0"` and
`pip_bootstrap.success: true`. All subprocess output is included for failures.

The earlier 1.0.0.0 installed experiment reproduced AppData redirection: the
logical interpreter failed to start; the physical interpreter with logical
script failed with a missing file; both physical paths succeeded. That
established the path-visibility failure. Version 1.1 exercises real pip setup
rather than repeating the failing controls.

The unpackaged frozen 1.1 control passed: pip was absent, bootstrap succeeded,
and pip 26.3 was importable from the diagnostic runtime. The installed MSIX 1.1
experiment also passed: AppData was redirected, the physical-path script ran,
pip was initially absent, and bootstrap installed pip 26.3 successfully into
the package-local diagnostic runtime. This validates bootstrap through pip;
full UWCC model installation still needs a rebuilt production package test.
The probe mirrors UWCC's full-trust manifest and
clears PyInstaller's DLL-directory/PATH additions for external Python launches.

## Cleanup after collecting the report

In the affected user's ordinary PowerShell:

```powershell
Get-AppxPackage -Name UWCC.StorePathProbe | Remove-AppxPackage
```

Package-local redirected data is removed by uninstall. Any unredirected data
may remain in `%LOCALAPPDATA%\UWCC.StorePathProbe`; delete only that diagnostic
folder if present. Keep a copy of the report first.

To remove the certificate trust, an administrator can run from the extracted
folder:

```powershell
$probeCertificate = Get-PfxCertificate -FilePath .\UWCC.StorePathProbe.cer
Remove-Item -LiteralPath ("Cert:\LocalMachine\TrustedPeople\" + $probeCertificate.Thumbprint)
```

## Rebuild locally

Use an extracted official Windows x64 embedded Python 3.14 distribution:

```powershell
.venv\Scripts\python.exe tools/build_store_path_probe.py --runtime <embedded-python-directory>
```

The builder uses PyInstaller, Pillow and Windows SDK MakeAppx. Output is under
`dist\store-path-probe`; signing must use a code-signing certificate whose
subject is `CN=UWCC Store Path Probe`. Only public certificates belong in the
transfer bundle. No production package, release version or Git tag is changed.
