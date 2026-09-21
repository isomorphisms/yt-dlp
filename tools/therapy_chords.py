#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import librosa
import numpy as np

PITCHES = np.array(["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"])

QUALITIES = {
    "":        {0: 1.00, 4: 0.90, 7: 0.85},
    "m":       {0: 1.00, 3: 0.90, 7: 0.85},
    "dim":     {0: 1.00, 3: 0.90, 6: 0.88},
    "7":       {0: 1.00, 4: 0.88, 7: 0.82, 10: 0.72},
    "maj7":    {0: 1.00, 4: 0.88, 7: 0.82, 11: 0.72},
    "m7":      {0: 1.00, 3: 0.88, 7: 0.82, 10: 0.72},
    "m6":      {0: 1.00, 3: 0.88, 7: 0.82, 9: 0.68},
}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def chord_templates():
    result = []
    for root in range(12):
        for suffix, degrees in QUALITIES.items():
            v = np.full(12, 0.04, dtype=float)
            chord_tones = set()
            for degree, weight in degrees.items():
                pc = (root + degree) % 12
                v[pc] = weight
                chord_tones.add(pc)
            result.append((root, suffix, unit(v), chord_tones))
    return result


TEMPLATES = chord_templates()


def score_chords(chroma, bass_pc):
    x = unit(chroma)
    scored = []
    for root, suffix, template, tones in TEMPLATES:
        score = float(np.dot(x, template))
        if bass_pc == root:
            score += 0.045
        elif bass_pc in tones:
            score += 0.018
        scored.append((score, root, suffix, tones))
    scored.sort(reverse=True)
    return scored


def label(root, suffix, tones, bass_pc):
    base = f"{PITCHES[root]}{suffix}"
    if bass_pc != root and bass_pc in tones:
        return f"{base}/{PITCHES[bass_pc]}"
    return base


def bass_pitch_class(cqt_low):
    energy = np.sum(cqt_low, axis=1)
    folded = np.zeros(12)
    for i, value in enumerate(energy):
        folded[i % 12] += value
    return int(np.argmax(folded)), folded


def collapse(rows):
    if not rows:
        return []
    groups = []
    start = 0
    current = rows[0]["label"]
    for i in range(1, len(rows) + 1):
        changed = i == len(rows) or rows[i]["label"] != current
        if changed:
            group = rows[start:i]
            duration = group[-1]["end"] - group[0]["start"]
            if duration >= 0.35:
                groups.append({
                    "start": group[0]["start"],
                    "end": group[-1]["end"],
                    "label": current,
                    "score": float(np.mean([r["score"] for r in group])),
                    "margin": float(np.mean([r["margin"] for r in group])),
                    "bass": max(
                        set(r["bass"] for r in group),
                        key=lambda p: sum(r["bass"] == p for r in group),
                    ),
                })
            start = i
            if i < len(rows):
                current = rows[i]["label"]
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("outdir")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    y, sr = librosa.load(args.audio, sr=22050, mono=True)
    duration = len(y) / sr

    y_harm, _ = librosa.effects.hpss(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    tempo = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=512)
    cqt_low = np.abs(librosa.cqt(
        y_harm,
        sr=sr,
        hop_length=512,
        fmin=librosa.note_to_hz("C1"),
        n_bins=36,
        bins_per_octave=12,
    ))

    if len(beat_frames) < 2:
        raise RuntimeError("beat tracker returned too few beats")

    rows = []
    for i in range(len(beat_frames) - 1):
        a = int(beat_frames[i])
        b = int(beat_frames[i + 1])
        if b <= a:
            continue
        c = np.mean(chroma[:, a:b], axis=1)
        bass_pc, _ = bass_pitch_class(cqt_low[:, a:b])
        scored = score_chords(c, bass_pc)
        best, second = scored[0], scored[1]
        best_score, root, suffix, tones = best
        rows.append({
            "start": float(beat_times[i]),
            "end": float(beat_times[i + 1]),
            "label": label(root, suffix, tones, bass_pc),
            "score": best_score,
            "margin": best_score - second[0],
            "bass": int(bass_pc),
            "alt": label(second[1], second[2], second[3], bass_pc),
            "alt_score": second[0],
        })

    groups = collapse(rows)

    with (outdir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write(f"duration_seconds: {duration:.3f}\n")
        f.write(f"detected_tempo_bpm: {tempo:.3f}\n")
        f.write(f"beats_analyzed: {len(rows)}\n")
        f.write("\nCollapsed independent chord guess\n")
        f.write("start-end  chord  bass  score  margin\n")
        for g in groups:
            f.write(
                f"{g['start']:7.2f}-{g['end']:7.2f}  "
                f"{g['label']:<10} {PITCHES[g['bass']]:<2} "
                f"{g['score']:.3f} {g['margin']:.3f}\n"
            )

    with (outdir / "beats.txt").open("w", encoding="utf-8") as f:
        f.write("start end best bass score margin alternative alt_score\n")
        for r in rows:
            f.write(
                f"{r['start']:.3f} {r['end']:.3f} {r['label']} "
                f"{PITCHES[r['bass']]} {r['score']:.5f} {r['margin']:.5f} "
                f"{r['alt']} {r['alt_score']:.5f}\n"
            )

    print((outdir / "summary.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
