from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Any

import jax
import jax.numpy as jnp
from flax import struct

if TYPE_CHECKING:
    from traj_align.floor.types import FloorAlignment


_JAX_EPSILON = 1e-9


@struct.dataclass
class PlaneDetectionConfig:
    distance_tolerance: float = 0.05
    normal_tolerance_degrees: float = 10.0
    min_inliers: int = 100
    min_inlier_ratio: float = 0.01
    max_planes: int = 12
    ransac_iterations: int = 800
    sample_points: int = 50_000
    structural_candidates: int = 6
    seed: int = 17
    vlm_enabled: bool = True
    #: Fail the build when ``vlm_enabled`` is set but the VLM never actually
    #: chose the floor plane. The geometry fallback is a *different ablation
    #: arm*: silently substituting it makes a run reported under a
    #: ``floor_vlm_*`` label report a number the VLM had no part in. An arm
    #: that wants the geometry result sets ``vlm_enabled: false`` instead.
    vlm_required: bool = True
    #: Which VLM answers:
    #:
    #: * ``"local"`` -- weights from ``traj_align/vlm/profiles/<vlm_local_profile>.yaml``,
    #:   loaded into this process with ``transformers`` and freed afterwards.
    #: * ``"served"`` -- any OpenAI-compatible ``/v1/chat/completions`` endpoint at
    #:   ``vlm_base_url`` (vLLM, ``ragmap-vlm-service``, or a hosted API), asking
    #:   for ``vlm_served_model``.
    #: * ``"openai"`` -- the paid OpenAI Responses API with ``vlm_model``.
    #:
    #: The choice is recorded in ``floor_alignment.json``.
    vlm_backend: str = "local"
    #: The remote checkpoint, used only when ``vlm_backend == "openai"``.
    vlm_model: str = "gpt-5.6"
    #: The bundled profile used when ``vlm_backend == "local"`` (and the default
    #: served model name). ``qwen3_vl_32b`` is RAGMAP's shipped floor profile;
    #: replaying 90 archived selections put agreement with ``gpt-5.6`` at 45.6%
    #: (2B), 61.1% (4B), 77.8% (8B) and 92.2% (32B, NF4), against a 27.8%
    #: constant-answer baseline. See ``traj_align.floor.vlm_images``.
    vlm_local_profile: str = "qwen3_vl_32b"
    #: The ``model`` field sent by the ``served`` backend, e.g.
    #: ``Qwen/Qwen3-VL-32B-Instruct`` on vLLM. ``None`` sends ``vlm_local_profile``,
    #: which is what ``ragmap-vlm-service`` expects.
    vlm_served_model: str | None = None
    #: How the ``served`` backend sends images: ``"base64"`` data URLs (works with
    #: any host) or ``"file"`` URLs (only when the server shares this filesystem).
    vlm_image_transport: str = "base64"
    #: For ``"openai"``, an explicit override of ``$OPENAI_BASE_URL`` (``None``
    #: means the SDK's own resolution); for ``"served"``, the endpoint itself
    #: (required), e.g. ``http://localhost:8000``.
    vlm_base_url: str | None = None
    #: Where to read the API key when ``OPENAI_API_KEY`` is unset. Used by
    #: ``"openai"``, and by ``"served"`` as an optional Bearer token.
    openai_api_key_file: str | None = None
    vlm_images_per_plane: int = 3
    vlm_max_observations: int = 180
    vlm_max_projected_points: int = 800
    #: How many detected planes get a contact sheet, and therefore how many the
    #: VLM is allowed to choose between. Split from ``structural_candidates``
    #: (which still bounds the *geometry* arm's window) on 2026-08-10, because the
    #: two answer different questions and 6 was wrong for this one.
    #:
    #: **Why 6 was wrong.** On office1 of the 2026-08-09 Replica campaign, all three
    #: profiles -- ``qwen3_vl_8b``, ``qwen3_vl_32b``, ``qwen3_5_35b_a3b`` -- picked
    #: the same plane, and reading that scene's contact sheets shows it is a large
    #: pale vertical partition with red points blanketing it, not a floor. Raising
    #: this to 12 and re-asking the identical question through
    #: ``scripts/floor-vlm-trial.py`` moved ``qwen3_vl_32b`` onto a plane whose
    #: points trace the dark walkable ground between the bins and the seating --
    #: rank **8** by structural score, so at 6 it had no sheet and could not have
    #: been named. No model size and no prompt can fix a candidate set that excludes
    #: the answer, which is why this is a config field and not a note.
    #:
    #: This is the whole basis for the default, and it is worth being clear about
    #: how thin it is: two human readings of one scene's sheets. It is not an
    #: accuracy measurement over the set, because there is no ground truth to
    #: measure against -- see ``traj_align.floor.vlm_images`` on why an up
    #: direction taken from the mapper's camera poses does not provide one.
    #:
    #: What raising it costs is the request, not the render: contact sheets are
    #: written per candidate (a few seconds each), and all of them go into *one*
    #: conversation, so 12 sheets is a 12-image question. That is the number to
    #: watch for VRAM and for the paid backend's token bill, and it is why this is
    #: bounded rather than simply set to ``max_planes``.
    vlm_candidate_planes: int = 12
    #: Cap on the *answer* to each floor question. Was hard-coded at 1000 in
    #: ``VlmRequest`` (matching what the Responses calls used) and is a field now
    #: only because ``vlm_thinking_enabled`` spends the same budget on reasoning.
    #: A one-line JSON object needs ~60 tokens; a Qwen3.5 deliberation over six
    #: contact sheets ran past 1000 on 7 of 8 scenes.
    vlm_answer_max_tokens: int = 1_000
    #: Let a hybrid-reasoning checkpoint think in text before it answers the floor
    #: questions. Off by default and only Qwen3.5 templates read it at all
    #: (``captioning_api.generate_conversations`` documents the mechanism).
    #:
    #: **Measured and not recommended on this question.** ``qwen3_5_35b_a3b`` with
    #: thinking on and ``vlm_answer_max_tokens: 4000`` produced 12301 characters of
    #: view-by-view deliberation on office0's six contact sheets and *still* never
    #: reached the JSON, at ~7 minutes for the one call against 5 s for
    #: ``qwen3_vl_32b``. At the default 1000 tokens it is worse (1/8 on the
    #: archived scenes, 7 of them with no JSON at all). The failure is not that the
    #: reasoning is bad -- it is correct and thorough -- but that a six-image
    #: comparison gives it more to say than any budget worth paying for, and
    #: ``vlm_required: true`` turns "ran out mid-thought" into a failed build.
    #:
    #: Use ``vlm_prompt_style: survey`` instead: same reasoning-before-answering,
    #: bounded because it is *inside* the answer's JSON, works on profiles with no
    #: reasoning template, and kept in the audit blob. See
    #: ``traj_align.floor.vlm_images.floor_selection_prompt``. This flag stays
    #: because it is the only way to see a Qwen3.5 profile's native reasoning at
    #: all, and because it now has to be *chosen* rather than inherited from a
    #: template default.
    vlm_thinking_enabled: bool = False
    #: Which wording of the floor question to ask.
    #:
    #: * ``"direct"`` -- the historical prompt, ending in "return only JSON like
    #:   ``{"plane_index": 0, ...}``". Every archived selection answered this one,
    #:   so it is the only wording that can be replayed against them.
    #: * ``"survey"`` -- the same question, but the answer's JSON leads with a
    #:   ``survey`` object naming what the red points sit on for *each* plane
    #:   before ``plane_index``. Greedy decoding means the survey is in context
    #:   when the index is emitted, so this is chain-of-thought that costs no
    #:   template support, survives into the audit blob, and is readable by a
    #:   human deciding whether to trust the pick.
    #:
    #: Kept as an enum rather than a free-form prompt override because the wording
    #: is a measured quantity here: see
    #: ``traj_align.floor.vlm_images.floor_selection_prompt``.
    vlm_prompt_style: str = "direct"
    #: How many turns the selection takes. **1 is one question**, down to the audit
    #: blob gaining no ``selection_turns`` or ``selection_scheme`` key, which is the
    #: point: the elimination scheme is an ablation arm to be compared against the
    #: single question, not a replacement for it.
    #:
    #: One question is not always one *call*: a reply that misreads the answer shape
    #: is re-asked with a correction up to ``_SELECTION_MAX_ATTEMPTS`` times, on both
    #: arms alike, so the arms stay comparable. ``vlm_result.attempts`` in
    #: ``selection.json`` records how many it took.
    #:
    #: ``R > 1`` means ``R - 1`` elimination turns and then one decisive selection
    #: turn. Each elimination turn shows the still-live contact sheets and asks the
    #: model to vote out the candidates least likely to be the floor; the final turn
    #: asks ``floor_selection_prompt``'s *unchanged* question of whatever survived.
    #: So the only thing this changes about the decision is who is on the ballot,
    #: which is what makes the two arms comparable at all.
    #:
    #: Why bother: a 12-candidate ballot (``vlm_candidate_planes``) is a 12-way
    #: comparison in one forward pass, and the failure it produces is not a wrong
    #: reason but a confident answer about the wrong plane. Three cheaper questions
    #: may be easier than one hard one -- may be, because at the time of writing this
    #: is a hypothesis with a trial harness (``scripts/floor-vlm-trial.py``) and no
    #: campaign behind it. That is also why the default is 1.
    #:
    #: **Trialled on 2026-08-10 over four Replica scenes and it changed no answer.**
    #: ``qwen3_vl_32b`` at 12 candidates picked the same plane at 1 and at 3 turns on
    #: office0/office1/office4/room2, and every winner was still on the final ballot,
    #: so the scheme demonstrated safety rather than benefit. What it voted out reads
    #: correctly from the sheets -- a ceiling, a partition beside a doorway, and the
    #: top *structural* candidate on room2, which is a large pale wall. Cost +18.6 s
    #: median per selection (~+25% of the step), not the ~2x the extra calls suggest,
    #: because the sheets are rendered once. The measured table is in this stage's
    #: profile and the per-scene records in ``notes/floor-vlm-trial-20260810.json``.
    #:
    #: What is *not* configurable, deliberately: the share each turn eliminates
    #: (30%) and how many times an unusable answer is re-asked (3). Both live as
    #: constants in ``traj_align.floor.vlm_images`` because they are the
    #: scheme's definition rather than per-arm knobs, and both are recorded into
    #: ``floor_plane_vlm/selection.json`` since nothing else records them. The count
    #: is derived from the *original* ballot, so every elimination turn drops the same
    #: integer, and the prompt states it as that integer rather than as a percentage.
    #:
    #: Three properties worth relying on:
    #:
    #: * At least two candidates always reach the final turn, so a ballot of 1 or 2
    #:   makes the scheme a no-op rather than a forced answer.
    #: * **An elimination turn cannot fail a build.** An unusable answer -- after its
    #:   retries -- keeps the whole ballot and is recorded; only the final turn
    #:   carries ``vlm_required`` semantics. Worst case the scheme costs two wasted
    #:   calls and degrades to the single-turn question.
    #: * Turn 1 carries every sheet, so the widest conversation, and therefore the
    #:   VRAM peak, is unchanged from a single-turn run. The added cost is two
    #:   smaller requests -- roughly 2x the per-selection call time at 12 candidates.
    #:
    #: The risk it introduces, stated plainly: an elimination turn can vote the real
    #: floor off the ballot, and the final answer will then be a well-reasoned choice
    #: among what is left. That is *harder* to diagnose than a single wrong pick, so
    #: ``selection.json`` records every turn's ballot and votes, and the way to judge
    #: an arm is to re-run the scene at 1 and at 3 and ask whether the single-turn
    #: pick survived elimination.
    vlm_selection_rounds: int = 1
    #: Whether a VLM-selected plane must survive the Open3D render-and-re-ask
    #: loop before it may be emitted. **Off by default**, which reverses the
    #: invariant that loop was written to enforce -- read
    #: ``traj_align.floor.open3d`` before turning it back on or leaving it
    #: off, because the reason is not that the check is worthless.
    #:
    #: What it costs when on: the loop renders three Open3D views of the whole
    #: map and asks the VLM "is this level, and if not which plane is", up to
    #: ``vlm_refinement_max_attempts`` times, and a run that never converges is a
    #: **hard build failure** (``FloorVerificationError``, since
    #: ``vlm_required``). Whether it converges turns out to depend on the model
    #: in a way nothing was measuring: the loop's success condition is the model
    #: *re-proposing the plane it was shown*, so a model that holds its opinion
    #: passes and a model that keeps suggesting alternatives exhausts the budget.
    #: Measured 2026-08-09 over the 8-scene Replica set, both arms first-pass
    #: selecting fine: ``qwen3_vl_8b`` converged on 8 of 8 (mostly at attempt 1,
    #: by repeating itself while still reporting ``floor_is_level: false``),
    #: ``qwen3_vl_32b`` converged on **1 of 8** -- the other 7 walked a different
    #: plane each attempt (e.g. 5 -> 10 -> 0) and died on
    #: ``max_attempts_exhausted``.
    #:
    #: So the check as written does not measure floor quality so much as model
    #: stubbornness, and it blocked every large-VLM arm this repo has tried. Off
    #: by default until that termination condition is reworked; the geometry is
    #: unaffected either way, since the rotation always comes from the selected
    #: plane's fitted normal and this loop only ever changes *which* plane.
    #:
    #: ``vlm_refinement_max_attempts: 0`` is the older way to say the same thing
    #: and still works; this flag exists because a count of zero is not a legible
    #: way to express a policy.
    open3d_verification_enabled: bool = False
    vlm_refinement_max_attempts: int = 3
    vlm_refinement_render_width: int = 1024
    vlm_refinement_render_height: int = 768
    #: Backoff for the two OpenAI Responses calls in
    #: ``traj_align.floor``. Every arm's floor step calls the same
    #: endpoint, so a matrix running ~130 jobs concurrently rate-limits itself:
    #: seven jobs died on ``RateLimitError`` in one campaign, each after paying
    #: for a full plane-detection pass. These are *whole-call* attempts --
    #: ``vlm_refinement_max_attempts`` above is a different thing entirely (how
    #: many times the VLM is asked to re-judge a render).
    #:
    #: Total attempts, not retries: 1 means no retry at all.
    vlm_retry_attempts: int = 6
    #: Multiplier for the jittered exponential wait, in seconds. Jitter matters
    #: more than the base here -- 130 workers that back off in lockstep just
    #: collide again on the next attempt.
    vlm_retry_initial_seconds: float = 2.0
    #: Ceiling for any single wait, including a server-sent ``Retry-After``.
    vlm_retry_max_seconds: float = 60.0
    wall_perpendicular_tolerance_degrees: float = 15.0
    #: Share of the off-plane points that must sit *below* the selected floor
    #: before the map is judged upside down and flipped 180 degrees about X.
    #:
    #: The plane fit gives a normal but not a side: ``n`` and ``-n`` describe the
    #: same surface, so a correct floor selection can still produce a map hanging
    #: underneath its own floor -- observed on Contempo, where the selected plane
    #: was right and behaved like a ceiling. Nothing about *which* plane is chosen
    #: is reconsidered here; only the sign of the normal, and only when the
    #: evidence is one-sided. 0.95 rather than a bare majority because a real
    #: floor in a partially mapped scene still has stray points beneath it
    #: (through-window returns, depth outliers, a stairwell), while a genuinely
    #: inverted map puts essentially the whole scene on one side.
    upright_below_fraction: float = 0.95

    def without_seed(self) -> "PlaneDetectionConfig":
        """Return the runtime configuration after its seed has been consumed."""

        return self.replace(seed=None)


@struct.dataclass
class Pose:
    """Position and xyzw orientation, represented entirely as JAX arrays."""

    translation: jax.Array
    rotation_xyzw: jax.Array

    @classmethod
    def from_dict(cls, value: Mapping[str, float]) -> "Pose":
        """Host/JSON boundary adapter."""

        return cls(
            translation=jnp.asarray((value["tx"], value["ty"], value["tz"]), dtype=float),
            rotation_xyzw=jnp.asarray((value["qx"], value["qy"], value["qz"], value["qw"]), dtype=float),
        )

    def to_dict(self) -> dict[str, float]:
        """Host/JSON boundary adapter."""

        return {
            "tx": float(self.translation[0]),
            "ty": float(self.translation[1]),
            "tz": float(self.translation[2]),
            "qx": float(self.rotation_xyzw[0]),
            "qy": float(self.rotation_xyzw[1]),
            "qz": float(self.rotation_xyzw[2]),
            "qw": float(self.rotation_xyzw[3]),
        }

    def transformed(self, alignment: "FloorAlignment") -> "Pose":
        """Return this pose expressed in a new map coordinate frame."""

        from traj_align.common.geometry import _multiply_quaternions_jax

        return self.replace(
            translation=alignment.transform_points(self.translation),
            rotation_xyzw=_multiply_quaternions_jax(
                alignment.rotation_xyzw, self.rotation_xyzw
            ),
        )

    @classmethod
    def stack(cls, poses: list["Pose"]) -> "Pose":
        """Build one batched pose PyTree from host-collected poses."""

        return cls(
            translation=jnp.stack([pose.translation for pose in poses]),
            rotation_xyzw=jnp.stack([pose.rotation_xyzw for pose in poses]),
        )


@struct.dataclass(eq=False)
class PlaneSummary:
    """A structural plane with JAX-resident numeric fields.

    ``eq=False`` is deliberate: planes contain arrays and callers identify a
    selected plane by its index in ``FloorAlignment.planes`` rather than by
    array equality.
    """

    index: jax.Array
    normal: jax.Array
    offset: jax.Array
    inliers: jax.Array
    support_ratio: jax.Array
    extent: jax.Array
    boundary_score: jax.Array
    structural_score: jax.Array

    def to_dict(self) -> dict[str, Any]:
        """Serialize the shared plane schema at an explicit host boundary."""

        return {
            "normal": [float(value) for value in self.normal],
            "offset": float(self.offset),
            "inliers": int(self.inliers),
            "support_ratio": float(self.support_ratio),
            "extent": [float(value) for value in self.extent],
            "boundary_score": float(self.boundary_score),
            "structural_score": float(self.structural_score),
        }


@struct.dataclass
class PlaneSet:
    """Fixed-capacity, structure-of-arrays representation of detected planes."""

    normals: jax.Array
    offsets: jax.Array
    inlier_counts: jax.Array
    support_ratios: jax.Array
    extents: jax.Array
    boundary_scores: jax.Array
    structural_scores: jax.Array
    valid: jax.Array

    def __len__(self) -> int:
        return int(jnp.sum(self.valid))

    def __getitem__(self, index: int | jax.Array) -> PlaneSummary:
        index = jnp.asarray(index, dtype=jnp.int32)
        return PlaneSummary(
            index=index,
            normal=self.normals[index],
            offset=self.offsets[index],
            inliers=self.inlier_counts[index],
            support_ratio=self.support_ratios[index],
            extent=self.extents[index],
            boundary_score=self.boundary_scores[index],
            structural_score=self.structural_scores[index],
        )

    def __iter__(self):
        # Iteration exists solely for host-side logging, VLM, and artifact I/O.
        return (self[index] for index in range(len(self)))

    def index(self, plane: PlaneSummary) -> int:
        return int(plane.index)

    def take(self, count: int) -> "PlaneSet":
        return self.replace(valid=self.valid & (jnp.arange(self.valid.shape[0]) < count))

    @staticmethod
    @jax.jit
    def from_ransac_state(points: jax.Array, state: Any, tolerance: float) -> "PlaneSet":
        """Summarize a fitted RANSAC state into its fixed-capacity plane set."""

        centroid = jnp.mean(points, axis=0)
        signs = jnp.where(state.normals @ centroid + state.offsets < 0.0, -1.0, 1.0)
        normals, offsets = state.normals * signs[:, None], state.offsets * signs
        masks = jnp.abs(points @ normals.T + offsets) <= tolerance
        masks = masks & state.valid[None, :]
        counts = masks.sum(axis=0)
        support = counts / points.shape[0]
        x_axis = jnp.array((1.0, 0.0, 0.0))
        y_axis = jnp.array((0.0, 1.0, 0.0))
        axes = jnp.where(jnp.abs(normals[:, :1]) > 0.9, y_axis, x_axis)
        basis_u = jnp.cross(normals, axes)
        basis_u = basis_u / jnp.maximum(jnp.linalg.norm(basis_u, axis=1, keepdims=True), _JAX_EPSILON)
        basis_v = jnp.cross(normals, basis_u)
        projected = jnp.stack((points @ basis_u.T, points @ basis_v.T), axis=-1)
        low = jnp.where(masks[..., None], projected, jnp.inf).min(axis=0)
        high = jnp.where(masks[..., None], projected, -jnp.inf).max(axis=0)
        extents = jnp.where(state.valid[:, None], high - low, 0.0)
        scene_radius = jnp.maximum(jnp.linalg.norm(points - centroid, axis=1).max(), _JAX_EPSILON)
        scene_diag = jnp.maximum(
            jnp.linalg.norm(points.max(axis=0) - points.min(axis=0)), _JAX_EPSILON
        )
        distances = jnp.linalg.norm(points - centroid, axis=1)[:, None]
        boundary = jnp.minimum(
            jnp.where(masks, distances, 0.0).max(axis=0) / scene_radius, 1.0
        )
        extent_scale = jnp.minimum(jnp.linalg.norm(extents, axis=1), scene_diag) / scene_diag
        structural = 0.45 * boundary + 0.35 * extent_scale + 0.20 * jnp.minimum(support * 4.0, 1.0)
        structural = jnp.where(state.valid, structural, -jnp.inf)
        order = jnp.argsort(structural)[::-1]
        return PlaneSet(
            normals=normals[order],
            offsets=offsets[order],
            inlier_counts=counts[order].astype(jnp.int32),
            support_ratios=support[order],
            extents=extents[order],
            boundary_scores=boundary[order],
            structural_scores=structural[order],
            valid=state.valid[order],
        )


@struct.dataclass
class WallAlignment:
    applied: jax.Array
    plane_index: jax.Array
    normal: jax.Array
    inliers: jax.Array
    yaw_degrees: jax.Array
    perpendicular_tolerance_degrees: jax.Array


@struct.dataclass
class MapCentering:
    source_centroid: jax.Array
    translation: jax.Array
