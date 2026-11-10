from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from roi3d.geometry import (
    V2_CACHE_SHAPE_ZYX, RECISTGeometry, geometry_from_mask,
    recist_geometries, sample_native,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the v2 memmapped lesion ROI cache")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _record(path: Path, root: Path, geometry: RECISTGeometry) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "source": path.parent.name,
        "lesion_id": geometry.lesion_id,
        "center_zyx": geometry.center_zyx.tolist(),
        "axis_xy": geometry.axis_xy.tolist(),
        "diameter_mm": geometry.diameter_mm,
        "diameter_true_mm": geometry.diameter_true_mm,
        "roi_spacing_mm": geometry.roi_spacing_mm,
    }


def scan_case(arguments: tuple[str, str]) -> list[dict]:
    path_string, root_string = arguments
    path, root = Path(path_string), Path(root_string)
    with np.load(path, allow_pickle=False) as case:
        spacing = np.asarray(case["spacing"], dtype=np.float64)
        recist = case["recist"] if "recist" in case.files else None
        if recist is not None and np.any(recist):
            geometries = recist_geometries(recist, spacing)
        else:
            # Only the handful of legacy cases without a RECIST raster need GT
            # decompression during indexing. The build pass will read GT once.
            gts = case["gts"]
            geometries = [geometry_from_mask(gts, int(i), spacing)
                          for i in np.unique(gts) if i != 0]
    return [_record(path, root, geometry) for geometry in geometries]


def create_manifest(data: Path, output: Path, workers: int) -> list[dict]:
    paths = sorted(data.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"no NPZ files below {data}")
    arguments = [(str(path), str(data)) for path in paths]
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case_records in pool.map(scan_case, arguments, chunksize=8):
            records.extend(case_records)
            if len(records) and len(records) % 1000 < len(case_records):
                print(f"indexed {len(records)} lesions", flush=True)
    for index, record in enumerate(records):
        record["index"] = index
    (output / "manifest.json").write_text(json.dumps(records))
    return records


def _geometry(record: dict) -> RECISTGeometry:
    return RECISTGeometry(
        lesion_id=int(record["lesion_id"]),
        center_zyx=np.asarray(record["center_zyx"], dtype=np.float64),
        axis_xy=np.asarray(record["axis_xy"], dtype=np.float64),
        diameter_mm=float(record["diameter_mm"]),
        diameter_true_mm=float(record["diameter_true_mm"]),
        roi_spacing_mm=float(record["roi_spacing_mm"]),
    )


def build_cases(arguments: tuple[str, str, list[dict]]) -> tuple[int, int]:
    data_string, output_string, records = arguments
    data, output = Path(data_string), Path(output_string)
    n = int(np.fromfile(output / "count.txt", dtype=np.int64)[0])
    shape = V2_CACHE_SHAPE_ZYX
    images = np.memmap(output / "images.uint8", mode="r+", dtype=np.uint8,
                       shape=(n, *shape))
    packed_masks = np.memmap(
        output / "masks.packbits", mode="r+", dtype=np.uint8,
        shape=(n, shape[0], shape[1], (shape[2] + 7) // 8),
    )
    done = np.memmap(output / "done.uint8", mode="r+", dtype=np.uint8, shape=(n,))
    pending = [record for record in records if not done[int(record["index"])]]
    if not pending:
        return len(records), 0
    completed = 0
    path = data / pending[0]["path"]
    with np.load(path, allow_pickle=False) as case:
        image = case["imgs"]
        gts = case["gts"]
        spacing = np.asarray(case["spacing"], dtype=np.float64)
    for record in pending:
        index = int(record["index"])
        geometry = _geometry(record)
        sampled_image = sample_native(
            image, geometry, spacing, shape, order=1
        )
        sampled_mask = sample_native(
            (gts == geometry.lesion_id).astype(np.uint8), geometry,
            spacing, shape, order=0,
        ).astype(bool)
        images[index] = np.clip(np.rint(sampled_image), 0, 255).astype(np.uint8)
        packed_masks[index] = np.packbits(sampled_mask, axis=-1)
        done[index] = 1
        completed += 1
    images.flush(); packed_masks.flush(); done.flush()
    return len(records), completed


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        records = json.loads(manifest_path.read_text())
    else:
        records = create_manifest(args.data, args.output, args.workers)
    n = len(records)
    np.asarray([n], dtype=np.int64).tofile(args.output / "count.txt")
    shape = V2_CACHE_SHAPE_ZYX
    files = (
        ("images.uint8", n * int(np.prod(shape))),
        ("masks.packbits", n * shape[0] * shape[1] * ((shape[2] + 7) // 8)),
        ("done.uint8", n),
    )
    for name, size in files:
        path = args.output / name
        if args.overwrite or not path.exists():
            with path.open("wb") as stream:
                stream.truncate(size)

    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["path"], []).append(record)
    tasks = [(str(args.data), str(args.output), group)
             for group in grouped.values()]
    processed = completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for case_count, new_count in pool.map(build_cases, tasks, chunksize=1):
            processed += case_count; completed += new_count
            if processed % 100 < case_count:
                print(f"visited {processed}/{n}; wrote {completed} new", flush=True)
    print(f"cache complete: {n} lesions, {completed} newly written")


if __name__ == "__main__":
    main()
