"""Evaluate upstream orientation models on ten equally spaced video samples.

Predetermined acceptance: >=8 of 10 samples must exceed the upstream strict
confidence threshold and agree on the same right-angle correction. This small
evaluation is not a general underwater accuracy certification.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from uwcc.media import probe, read_frame

SPECS = {
    "fast": ("fast-s224-fp32.onnx", 224, .738,
             "2a61ecc83cf27f5869a4a806c7e87ade81608f7356f2b7f2d6de4c8c38c32a0d"),
    "max": ("max-l280-int8.onnx", 280, .711,
            "9e358f9c51587ec4472ac9496e52dbcaa428886613ebe338fb0654b10bdc3ccf"),
}


def tensor(image, size):
    factor = size/max(image.size)
    width, height = (max(1, round(side*factor)) for side in image.size)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    scaled = image.resize((width, height), Image.Resampling.BICUBIC)
    canvas.paste(scaled, ((size-width)//2, (size-height)//2))
    pixels = np.asarray(canvas, dtype=np.float32)/255
    pixels -= np.array([.485, .456, .406], dtype=np.float32)
    pixels /= np.array([.229, .224, .225], dtype=np.float32)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def prediction(probability):
    probability = probability[0]
    peak = int(np.argmax(probability))
    offsets = np.arange(-10, 11)
    weights = probability[(peak+offsets) % 360]
    angle = float((peak+np.dot(weights, offsets)/max(float(weights.sum()), 1e-12)) % 360)
    distances = np.abs((np.arange(360)-angle+180) % 360-180)
    confidence = float(probability[distances <= 10].sum())
    correction = (-int((angle+45)//90)*90) % 360
    sectors = ((np.arange(360)+45)//90) % 4
    quarter = {str((-index*90) % 360): float(probability[sectors == index].sum()) for index in range(4)}
    best = max(quarter, key=quarter.get)
    return {"angle_cw": angle, "confidence": confidence, "correction_cw": correction,
            "quarter_probabilities": quarter, "quarter_correction_cw": int(best),
            "quarter_confidence": quarter[best]}


def main():
    import onnxruntime as ort
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=Path("models/rightwayup"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results/orientation"))
    parser.add_argument("--tier", choices=["fast", "max", "both"], default="both")
    parser.add_argument("--controls", action="store_true", help="Also evaluate 90/180/270-degree rotated samples")
    parser.add_argument("--decision", choices=["strict", "quarter"], default="strict",
                        help="Quarter rule: >=8/10 votes with sector probability >=0.9, plus mean winner probability >=0.9")
    args = parser.parse_args()
    # Verified visually using upright rotated previews, not inferred from predictions.
    expected = {"C0201": 0, "C0202": 270, "C0212": 90, "C0213": 90}
    samples = {}
    for clip, correction in expected.items():
        info = probe(args.directory/(clip+".MP4"))
        folder = args.output_dir/"frames"/clip
        folder.mkdir(parents=True, exist_ok=True)
        samples[clip] = []
        for index in range(10):
            seconds = info.duration*(index+.5)/10
            file = folder/f"{index:02}.png"
            if not file.is_file():
                frame = read_frame(info, seconds, max_width=640)
                Image.fromarray(np.rint(np.clip(frame, 0, 1)*255).astype(np.uint8)).save(file)
            samples[clip].append((seconds, file))
        print(f"Sampled {clip}: expected correction {correction}° CW", flush=True)
    ort.disable_telemetry_events()
    results = []
    for tier, (filename, size, threshold, checksum) in SPECS.items():
        if args.tier not in (tier, "both"):
            continue
        path = args.models_dir/filename
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Model checksum mismatch: {filename}")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        for clip, frames in samples.items():
            for turn in ((0, 90, 180, 270) if args.controls else (0,)):
                measurements = []
                start = perf_counter()
                for seconds, file in frames:
                    with Image.open(file) as opened:
                        image = opened.convert("RGB")
                    if turn:
                        image = image.transpose({90: Image.Transpose.ROTATE_270,
                                                 180: Image.Transpose.ROTATE_180,
                                                 270: Image.Transpose.ROTATE_90}[turn])
                    prob = session.run(None, {"image": tensor(image, size)})[0]
                    measurements.append({"seconds": seconds, **prediction(prob)})
                if args.decision == "quarter":
                    confident = [item for item in measurements if item["quarter_confidence"] >= .9]
                    votes = Counter(item["quarter_correction_cw"] for item in confident)
                else:
                    confident = [item for item in measurements if item["confidence"] >= threshold]
                    votes = Counter(item["correction_cw"] for item in confident)
                winner, count = votes.most_common(1)[0] if votes else (None, 0)
                accepted = winner if count >= 8 else None
                mean_probability = float(np.mean([item["quarter_probabilities"].get(str(winner), 0) for item in measurements]))
                if args.decision == "quarter" and mean_probability < .9:
                    accepted = None
                target = (expected[clip]-turn) % 360
                record = {"tier": tier, "clip": clip, "synthetic_rotation_cw": turn,
                          "expected_correction_cw": target, "accepted_correction_cw": accepted,
                          "strict_threshold": threshold, "confident_votes": dict(votes),
                          "decision": args.decision, "mean_winner_probability": mean_probability,
                          "all_votes": dict(Counter(item["correction_cw"] for item in measurements)),
                          "correct": accepted == target, "seconds": perf_counter()-start,
                          "samples": measurements}
                results.append(record)
                print(f"{tier} {clip} +{turn}°: target={target}°, accepted={accepted}, "
                      f"confident votes={dict(votes)}, all={record['all_votes']} · {record['seconds']:.2f}s", flush=True)
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir/f"evaluation-{args.tier}-{args.decision}{'-controls' if args.controls else ''}.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
