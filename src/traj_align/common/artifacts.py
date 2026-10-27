"""Rigid-transform application and aligned artifact readers/writers."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable
import jax
import jax.numpy as jnp
import numpy as np
from traj_align.utils.jax_batched_map import DEFAULT_BATCH_SIZE as _JAX_BATCH_SIZE, iter_batches
from traj_align.common.types import Pose
from traj_align.common.transforms import PlanarTransform, RigidTransform, transform_pose_dict as _transform_planar_pose

_HOST_IO_BATCH_SIZE = 65_536

if TYPE_CHECKING:
    from traj_align.floor.types import FloorAlignment


@jax.jit
def _transform_points_batch_jax(points: jax.Array, alignment: FloorAlignment) -> jax.Array:
    """Apply one reusable fixed-size point-transform kernel."""

    return alignment.transform_points(points)


def _transform_host_points(points: np.ndarray, alignment: FloorAlignment | PlanarTransform | RigidTransform) -> np.ndarray:
    """Transform host points in fixed JIT batches without a full-cloud upload.

    ``iter_batches`` handles an empty ``points`` array itself (one empty
    batch, no padding), so there is no separate empty-input guard here.
    """

    if isinstance(alignment, (PlanarTransform, RigidTransform)):
        return alignment.transform_points(points)
    transformed = [
        np.asarray(jax.device_get(batch))
        for batch in iter_batches(
            _transform_points_batch_jax,
            points,
            alignment,
            in_axes=(0, None),
            batch_size=_JAX_BATCH_SIZE,
        )
    ]
    return np.concatenate(transformed, axis=0)


def transform_point(point: Iterable[float], alignment: FloorAlignment | PlanarTransform | RigidTransform) -> tuple[float, float, float]:
    """Host-facing adapter for single point/file-record call sites."""

    if isinstance(alignment, (PlanarTransform, RigidTransform)):
        transformed = alignment.transform_points(np.asarray((tuple(point),), dtype=float))[0]
        return tuple(float(value) for value in transformed)  # type: ignore[return-value]
    transformed = alignment.transform_points(jnp.asarray(tuple(point), dtype=float))
    return tuple(float(value) for value in transformed)  # type: ignore[return-value]


def transform_pose_dict(pose: dict[str, Any], alignment: FloorAlignment | PlanarTransform | RigidTransform) -> dict[str, float]:
    """JSON adapter around :meth:`FloorAlignment.transform_pose`."""

    if isinstance(alignment, RigidTransform):
        return alignment.transform_pose_dict(pose)
    if isinstance(alignment, PlanarTransform):
        return _transform_planar_pose(pose, alignment)
    return alignment.transform_pose(Pose.from_dict(pose)).to_dict()


#: Observations transformed and written per batch.
#:
#: This function used to hold every payload in memory at once. That is fine for
#: a pose -- 80 bytes -- but an observation record is dominated by its ORB
#: ``features``: 156 KiB of the 158 KiB a TorWIC record measures, none of which
#: this function touches. At Replica's 2000 frames that is ~300 MiB of parsed
#: JSON and the step fits its measured 4096 MiB; at TorWIC's 6204 it is ~1 GiB
#: of text expanded into Python objects, and the step was OOM-killed twice
#: (allocation 10508342, steps 78 and 137) with the VLM selection already
#: finished and only the export left to do.
#:
#: Batched rather than one-at-a-time because ``FloorAlignment.transform_poses``
#: is a JAX call: per-record dispatch would trade the memory for latency across
#: thousands of tiny calls. 512 records is ~80 MiB of peak payload, two orders
#: below the limit, while keeping the transform vectorised.
_OBSERVATION_BATCH = 512


def write_aligned_observations(source: Path, destination: Path, alignment: FloorAlignment | PlanarTransform | RigidTransform) -> None:
    """Copy ``source`` to ``destination`` with every pose moved into the aligned frame.

    Streams in batches, so peak memory is set by ``_OBSERVATION_BATCH`` rather
    than by the trajectory's length. Written straight to ``destination`` rather
    than to a temporary: ``stages.floor`` line-counts this file while it grows
    to report per-observation progress, which a rename at the end would flatten
    to nothing until completion.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)

    def flush(output, payloads: list, pose_locations: list[int], poses: list) -> None:
        if isinstance(alignment, (PlanarTransform, RigidTransform)):
            for payload_index in pose_locations:
                payloads[payload_index]["pose"] = transform_pose_dict(
                    payloads[payload_index]["pose"], alignment
                )
        elif poses:
            transformed = alignment.transform_poses(Pose.stack(poses))
            for index, payload_index in enumerate(pose_locations):
                payloads[payload_index]["pose"] = Pose(
                    translation=transformed.translation[index],
                    rotation_xyzw=transformed.rotation_xyzw[index],
                ).to_dict()
        for payload in payloads:
            output.write(json.dumps(payload, sort_keys=True) + "\n")

    with source.open("r", encoding="utf-8") as input_, destination.open("w", encoding="utf-8") as output:
        payloads: list = []
        pose_locations: list[int] = []
        poses: list = []
        for raw_line in input_:
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            pose = payload.get("pose")
            if pose is not None:
                pose_locations.append(len(payloads))
                poses.append(Pose.from_dict(pose))
            payloads.append(payload)
            if len(payloads) >= _OBSERVATION_BATCH:
                flush(output, payloads, pose_locations, poses)
                payloads, pose_locations, poses = [], [], []
        flush(output, payloads, pose_locations, poses)


def write_aligned_tum_trajectory(source: Path, destination: Path, alignment: FloorAlignment | PlanarTransform | RigidTransform) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_lines = []
    timestamps = []
    poses = []
    with source.open("r", encoding="utf-8") as input_:
        for raw_line in input_:
            line = raw_line.strip()
            if not line:
                output_lines.append(raw_line)
                continue
            if line.startswith("#"):
                output_lines.append(raw_line)
                continue
            timestamp, tx, ty, tz, qx, qy, qz, qw = (float(value) for value in line.split())
            timestamps.append(timestamp)
            poses.append(Pose(
                translation=jnp.asarray((tx, ty, tz)), rotation_xyzw=jnp.asarray((qx, qy, qz, qw))
            ))
            output_lines.append(None)

    if isinstance(alignment, (PlanarTransform, RigidTransform)):
        pose_index = 0
        for index, line in enumerate(output_lines):
            if line is not None:
                continue
            transformed = transform_pose_dict({
                "tx": poses[pose_index].translation[0], "ty": poses[pose_index].translation[1],
                "tz": poses[pose_index].translation[2], "qx": poses[pose_index].rotation_xyzw[0],
                "qy": poses[pose_index].rotation_xyzw[1], "qz": poses[pose_index].rotation_xyzw[2],
                "qw": poses[pose_index].rotation_xyzw[3],
            }, alignment)
            output_lines[index] = (
                f"{timestamps[pose_index]:.9f} {transformed['tx']:.9f} {transformed['ty']:.9f} "
                f"{transformed['tz']:.9f} {transformed['qx']:.9f} {transformed['qy']:.9f} "
                f"{transformed['qz']:.9f} {transformed['qw']:.9f}\n"
            )
            pose_index += 1
    elif poses:
        transformed = alignment.transform_poses(Pose.stack(poses))
        pose_index = 0
        for index, line in enumerate(output_lines):
            if line is not None:
                continue
            pose = Pose(
                translation=transformed.translation[pose_index], rotation_xyzw=transformed.rotation_xyzw[pose_index]
            )
            output_lines[index] = (
                f"{timestamps[pose_index]:.9f} {float(pose.translation[0]):.9f} {float(pose.translation[1]):.9f} "
                f"{float(pose.translation[2]):.9f} {float(pose.rotation_xyzw[0]):.9f} "
                f"{float(pose.rotation_xyzw[1]):.9f} {float(pose.rotation_xyzw[2]):.9f} "
                f"{float(pose.rotation_xyzw[3]):.9f}\n"
            )
            pose_index += 1

    with destination.open("w", encoding="utf-8") as output:
        output.writelines(output_lines)


def write_aligned_map_points_csv(source: Path, destination: Path, alignment: FloorAlignment | PlanarTransform | RigidTransform) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", newline="") as input_:
        reader = csv.DictReader(input_)
        fieldnames = reader.fieldnames or ["id", "x", "y", "z", "observations", "found", "descriptor_hex"]
        rows = list(reader)
    points = np.asarray([(row["x"], row["y"], row["z"]) for row in rows], dtype=float)
    transformed = _transform_host_points(points, alignment)
    with destination.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row, (x, y, z) in zip(rows, transformed, strict=True):
            row.update({"x": f"{x:.9f}", "y": f"{y:.9f}", "z": f"{z:.9f}"})
            writer.writerow(row)


def write_aligned_ply_pointcloud(source: Path, destination: Path, alignment: FloorAlignment | PlanarTransform | RigidTransform) -> None:
    header, payload = _read_ply(source)
    if header["format"] == "binary_little_endian":
        _write_aligned_binary_ply(destination, header, payload, alignment)
        return
    if header["format"] == "ascii":
        _write_aligned_ascii_ply(destination, header, payload, alignment)
        return
    raise ValueError(f"{source}: unsupported PLY format {header['format']!r}")


def _load_detection_points(*, pointcloud: Path | None, map_points: Path | None) -> tuple[np.ndarray, Path | None]:
    if pointcloud and pointcloud.is_file():
        try:
            header, payload = _read_ply(pointcloud)
            return _points_from_ply(header, payload), pointcloud
        except ValueError:
            pass
    if map_points and map_points.is_file():
        points = []
        with map_points.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                points.append((float(row["x"]), float(row["y"]), float(row["z"])))
        return np.asarray(points, dtype=float), map_points
    return np.empty((0, 3), dtype=float), None


def _aligned_sibling(path: Path | None, marker: str) -> Path | None:
    if path is None:
        return None
    return path.with_name(f"{path.stem}.{marker}{path.suffix}")


def _read_ply(path: Path) -> tuple[dict[str, Any], bytes]:
    header_lines: list[str] = []
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"{path}: incomplete PLY header")
            line = raw.decode("ascii", errors="replace").rstrip("\n")
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        payload = handle.read()

    if not header_lines or header_lines[0].strip() != "ply":
        raise ValueError(f"{path}: not a PLY file")

    vertex_count: int | None = None
    format_name: str | None = None
    properties: list[tuple[str, str]] = []
    in_vertex = False
    for line in header_lines:
        fields = line.strip().split()
        if not fields:
            continue
        if fields[:1] == ["format"]:
            format_name = fields[1]
        elif fields[:2] == ["element", "vertex"]:
            vertex_count = int(fields[2])
            in_vertex = True
        elif fields[:1] == ["element"]:
            in_vertex = False
        elif in_vertex and fields[:1] == ["property"]:
            properties.append((fields[1], fields[2]))

    if vertex_count is None or format_name is None:
        raise ValueError(f"{path}: missing PLY vertex count or format")
    return {
        "path": path,
        "lines": header_lines,
        "format": format_name,
        "vertex_count": vertex_count,
        "properties": properties,
    }, payload


def _points_from_ply(header: dict[str, Any], payload: bytes) -> np.ndarray:
    if header["format"] == "binary_little_endian":
        data = np.frombuffer(payload, dtype=_ply_dtype(header), count=header["vertex_count"])
        return np.column_stack([data["x"], data["y"], data["z"]]).astype(float)
    if header["format"] == "ascii":
        rows = payload.decode("ascii", errors="replace").splitlines()[: header["vertex_count"]]
        indices = _ply_xyz_indices(header)
        return np.asarray([[float(row.split()[index]) for index in indices] for row in rows], dtype=float)
    raise ValueError(f"{header['path']}: unsupported PLY format {header['format']!r}")


def _write_aligned_binary_ply(destination: Path, header: dict[str, Any], payload: bytes, alignment: FloorAlignment | PlanarTransform) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dtype = _ply_dtype(header)
    data = np.frombuffer(payload, dtype=dtype, count=header["vertex_count"]).copy()
    _transform_ply_coordinates_in_place(data, alignment)
    trailing_payload = payload[dtype.itemsize * header["vertex_count"] :]
    header_bytes = ("\n".join(header["lines"]) + "\n").encode("ascii")
    destination.write_bytes(header_bytes + data.tobytes() + trailing_payload)


def _transform_ply_coordinates_in_place(data: np.ndarray, alignment: FloorAlignment | PlanarTransform) -> None:
    """Apply the rigid transform to a binary PLY in bounded JAX batches.

    Replica point clouds can contain tens of millions of vertices.  Host I/O
    uses larger chunks, while the device helper splits each one into padded
    2,048-point kernels so changing point-cloud sizes does not recompile JAX.
    """

    for start in range(0, len(data), _HOST_IO_BATCH_SIZE):
        chunk = data[start : start + _HOST_IO_BATCH_SIZE]
        points = np.column_stack((chunk["x"], chunk["y"], chunk["z"])).astype(np.float64, copy=False)
        transformed = _transform_host_points(points, alignment)
        chunk["x"], chunk["y"], chunk["z"] = transformed.T


def _transform_ply_coordinates(data: np.ndarray, alignment: FloorAlignment | PlanarTransform) -> np.ndarray:
    """Backward-compatible in-place PLY coordinate transformation helper."""

    _transform_ply_coordinates_in_place(data, alignment)
    return data


def _write_aligned_ascii_ply(destination: Path, header: dict[str, Any], payload: bytes, alignment: FloorAlignment | PlanarTransform) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.decode("ascii", errors="replace").splitlines()
    x_index, y_index, z_index = _ply_xyz_indices(header)
    vertex_rows = rows[: header["vertex_count"]]
    values_by_row = [row.split() for row in vertex_rows]
    points = np.asarray(
        [[float(values[index]) for index in (x_index, y_index, z_index)] for values in values_by_row],
        dtype=float,
    )
    transformed = _transform_host_points(points, alignment)
    output_rows = []
    for values, (x, y, z) in zip(values_by_row, transformed, strict=True):
        values[x_index] = f"{x:.9f}"
        values[y_index] = f"{y:.9f}"
        values[z_index] = f"{z:.9f}"
        output_rows.append(" ".join(values))
    output_rows.extend(rows[header["vertex_count"] :])
    destination.write_text("\n".join(header["lines"] + output_rows) + "\n", encoding="ascii")


def _ply_dtype(header: dict[str, Any]) -> np.dtype:
    type_map = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int16": "<i2",
        "uint16": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "int32": "<i4",
        "uint32": "<u4",
        "float": "<f4",
        "float32": "<f4",
        "double": "<f8",
        "float64": "<f8",
    }
    fields = []
    for kind, name in header["properties"]:
        if kind not in type_map:
            raise ValueError(f"{header['path']}: unsupported PLY property type {kind!r}")
        fields.append((name, type_map[kind]))
    names = {name for _, name in header["properties"]}
    if not {"x", "y", "z"}.issubset(names):
        raise ValueError(f"{header['path']}: PLY vertices must include x, y, z")
    return np.dtype(fields)


def _ply_xyz_indices(header: dict[str, Any]) -> tuple[int, int, int]:
    names = [name for _, name in header["properties"]]
    return names.index("x"), names.index("y"), names.index("z")
