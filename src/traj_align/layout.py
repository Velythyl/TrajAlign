"""Output filenames written by floor alignment."""

#: Per-trajectory floor-alignment metadata/audit file.
FLOOR_ALIGNMENT_METADATA = "floor_alignment.json"

#: The filename marker floor alignment inserts before an artifact's suffix:
#: ``map_points.csv`` -> ``map_points.floor_aligned.csv``.
FLOOR_ALIGNED_MARKER = "floor_aligned"

#: Floor-aligned copy of the input ``observations.jsonl``.
FLOOR_ALIGNED_OBSERVATIONS = f"observations.{FLOOR_ALIGNED_MARKER}.jsonl"
