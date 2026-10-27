"""Persistent batch state, with interrupted jobs paused on restart."""

from __future__ import annotations

import uuid
from pathlib import Path

from .analysis import analyze
from .errors import Cancelled, UWCCError
from .media import probe
from .project import Grade, Project, atomic_json, fingerprint, read_json
from .render import RENDER_PROFILES, render


class BatchQueue:
    def __init__(self, path):
        self.path = Path(path)
        self.jobs = []
        if self.path.exists():
            data = read_json(self.path)
            self.jobs = data.get("jobs", [])
            for job in self.jobs:
                if job["state"] in ("Analyzing", "Rendering", "Waiting"):
                    job["state"] = "Paused"

    def save(self):
        atomic_json(self.path, {"schema_version": 1, "jobs": self.jobs})

    def add(self, source, folder, grade, profile):
        source, folder = Path(source).resolve(), Path(folder).resolve()
        extension = RENDER_PROFILES[profile][1]
        dest = folder / f"{source.stem}_corrected{extension}"
        reserved = {job["destination"] for job in self.jobs}
        number = 2
        while str(dest) in reserved or dest.exists() or dest == source:
            dest = folder / f"{source.stem}_corrected_{number}{extension}"
            number += 1
        job = {"id": str(uuid.uuid4()), "source": str(source), "destination": str(dest),
               "grade": grade.to_dict(), "profile": profile, "state": "Waiting", "message": "", "progress": 0}
        self.jobs.append(job)
        self.save()
        return job

    def process(self, cancel, progress=lambda *_: None):
        for job in self.jobs:
            if cancel.is_set():
                break
            if job["state"] not in ("Waiting", "Paused", "Failed", "Canceled"):
                continue

            def report(fraction, message):
                job["progress"], job["message"] = fraction, message
                progress(fraction, f"{Path(job['source']).name}: {message}")

            try:
                grade = Grade.from_dict(job["grade"])
                # A local model path is an explicit user selection in this local queue.
                grade.model_manifest = job["grade"].get("model_manifest")
                grade.denoise_manifest = job["grade"].get("denoise_manifest") or grade.denoise_manifest
                info = probe(job["source"], cancel)
                if grade.input_profile == "auto":
                    grade.input_profile = info.profile
                if grade.input_profile == "auto":
                    raise UWCCError("Input profile is ambiguous. " + info.profile_evidence + " Select an input profile for this clip.")
                project = Project(info.path, grade, fingerprint(Path(info.path)))
                job["state"] = "Analyzing"
                self.save()
                from .orientation import project_info
                info = project_info(project, info, cancel, report)
                project.analysis = analyze(info, grade, cancel, report)
                job["state"] = "Rendering"
                self.save()
                render(project, job["destination"], job["profile"], cancel, report)
                job["state"], job["progress"] = "Complete", 1
            except Cancelled:
                job["state"], job["message"] = "Paused", "Paused; resume restarts this item safely."
            except Exception as exc:
                job["state"], job["message"] = "Failed", str(exc)
            finally:
                self.save()
                progress(job["progress"], job["message"])
        return self.jobs
