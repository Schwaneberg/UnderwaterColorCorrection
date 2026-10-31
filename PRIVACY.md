# Privacy Policy for Underwater Color Correction (UWCC)

**Last updated:** October 31, 2026

This Privacy Policy describes how **Underwater Color Correction (UWCC)** ("we", "us", or "the application") handles information when you use our desktop application and command-line tools.

## Summary

* **No Personal Data Collected:** UWCC does not collect, store, transmit, sell, or share personal information.
* **Local-First Processing:** All video decoding, color analysis, stabilization, rendering, and project saving occur entirely on your local computer.
* **No Telemetry or Tracking:** The application contains no analytics, telemetry, advertising, or usage-tracking software.
* **No Accounts Required:** The software does not require registration, sign-in, or account creation.

---

## 1. Local Processing of Media and Files

UWCC operates locally on your device. Video footage, exported files, project files (`.json`), and custom presets are stored exclusively on your local or external drives. Neither the application nor its developers have access to your video content, filenames, or filesystem data.

---

## 2. Network Activity and Optional Features

UWCC is designed to operate completely offline. The application performs network requests **only** when explicitly initiated by the user for the following optional features:

### A. Optional AI Model Downloads
If you choose to download optional neural enhancement or denoise models (via **AI models…** in the desktop interface or `uwcc model-tools`), the application establishes direct HTTPS connections to upstream public repositories (such as GitHub and Hugging Face) to download model source code and verified weights. No personal data, device identifiers, or user tracking tokens are transmitted during these requests.

### B. Optional Cloud Grading Advice
If you explicitly configure and invoke the optional cloud advice feature (`uwcc advise` or through the desktop UI):
* Three downscaled proxy frames and current grading parameters are sent directly to the third-party vision API provider you configure (such as OpenAI or Anthropic).
* Requests are authenticated using your personal API key provided via your local environment variables.
* No proxy frames, project data, or API keys are ever routed through or stored on any server operated by the UWCC developers.
* Interactions with third-party API providers are governed by their respective privacy policies and terms of service.

---

## 3. Data Retention and Deletion

Because UWCC does not collect or store user data on remote servers, there is no remote user data to retain or delete. All application configuration files and downloaded model weights reside in standard local user application directories (such as `%LOCALAPPDATA%\uwcc` on Windows or `~/.local/share/uwcc` on Linux) and can be deleted by the user at any time.

---

## 4. Children's Privacy

UWCC does not knowingly collect or solicit personal information from children or anyone else.

---

## 5. Changes to This Policy

If this Privacy Policy is updated, changes will be published directly in this repository with a revised "Last updated" date.

---

## 6. Contact

If you have questions or feedback regarding this Privacy Policy, please open an issue in the project's official GitHub repository:
https://github.com/Schwaneberg/UnderwaterColorCorrection/issues
