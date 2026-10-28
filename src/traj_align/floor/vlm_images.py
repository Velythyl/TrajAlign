"""VLM contact-sheet generation and semantic floor selection.

Which plane this module picks decides the map's whole frame: the floor normal is
rotated onto +Z and that transform is applied to the observations, both
trajectories and the point cloud, so every embedding and every retrieval result
downstream inherits it. A wrong pick does not fail loudly -- it publishes a
plausible map in the wrong frame. That is why the model behind it is measured
rather than assumed.

One turn or three
-----------------

``PlaneDetectionConfig.vlm_selection_rounds`` decides the *shape* of the question.
At 1 -- the shipped default -- every contact sheet goes into one conversation and
the model names a ``plane_index``, which is what every measurement below was taken
against. Above 1 the same sheets are used to run elimination turns first: the model
votes candidates off the ballot, ``_elimination_count`` of them per turn (30% of the
original ballot, expressed to the model as an integer), and the final turn asks
``floor_selection_prompt``'s *unchanged* question of whatever survived. At the
shipped twelve candidates that is 12 -> 8 -> 4.

The argument for it is that the single call is a twelve-way comparison in one
forward pass, and its failure mode is a confident answer about the wrong plane
rather than a visibly bad one. The argument for keeping it off by default is
measured: over office0, office1, office4 and room2 on 2026-08-10, ``qwen3_vl_32b``
chose the **same plane at 1 turn and at 3**, with every winner still on the final
ballot, for +18.6 s median. So the scheme is safe on that sample and buys nothing
on it. The eliminations themselves were sound -- a ceiling, a partition beside a
doorway, and room2's top *structural* candidate, which is a large pale wall -- which
is the encouraging part and not the same as a benefit.

Note what that comparison is, because it is the only honest one available here:
"did the single-turn pick survive elimination on identical sheets", not "does the
three-turn answer look right". A turn that votes the floor out leaves the final turn
choosing sensibly among the rest, which reads exactly like success.

Two properties are structural rather than measured, and both matter more than the
hypothesis:

* **An elimination turn cannot fail a build.** An unusable vote is re-asked up to
  ``_ELIMINATION_MAX_ATTEMPTS`` times with a clause naming what was wrong, and then
  keeps the entire ballot. Only the final turn carries ``vlm_required``. So the
  worst case is two wasted calls and the single-turn question.

  The final turn has the same repair loop (``_SELECTION_MAX_ATTEMPTS``), and both
  arms get it: at ``rounds: 1`` the single question *is* the final turn. So "the
  model misread the answer shape" is no longer a way to fail a build in one call --
  which it was until 2026-08-12, when it failed one. What neither loop retries is a
  *decision*: a model that names a real but wrong plane has answered.
* **An eliminated plane is gone.** Nothing later can restore it, so the prompt tells
  the model to name fewer when it is unsure and ``_sanitise_elimination_votes``
  honours that. This is also why the "normals do not encode gravity" warning is in
  *both* elimination styles rather than only in ``survey``.

What can and cannot be used as ground truth here
------------------------------------------------

Every accuracy figure in this module is *agreement* with some other chooser --
``gpt-5.6`` while the archive held its selections, ``qwen3_vl_8b`` after that
archive was pruned. It would be much better to know whether the selected plane
**is** the floor, and one route to that looks obvious and is closed:

**Do not derive an up direction from the mapper's camera poses.** ORB-SLAM3's
poses are self-consistent along a trajectory and carry no fixed relationship to the
world frame; their gauge is arbitrary, which is precisely why this stage exists. A
plane's angle against a "camera up" is therefore an angle in a frame nobody has
fixed yet. It was tried on 2026-08-10 -- both as a prompt hint and as an accuracy
oracle -- and reverted the same day for that reason. Anything it appeared to prove
is not evidence.

What *is* available without gravity:

* **The contact sheets, read by a human.** Slow and the only real arbiter. It is
  how office1's failure was identified: the plane every profile picked has red
  points blanketing a large pale vertical partition, while the plane it picks with
  ``vlm_candidate_planes: 12`` traces the dark walkable ground between the bins and
  the seating.
* **Frame-free geometry.** A plane with a third of the map's points beneath it is
  not the floor whatever the orientation, and two candidates whose normals match up
  to sign are parallel. Both facts survive an arbitrary gauge; "which side is up"
  does not.

``scripts/floor-vlm-trial.py`` re-runs one finished build's floor decision -- real
plane detection, real sheets, one real VLM call -- in about a minute, so a config
change can be compared on identical evidence without a pipeline rebuild. It
reports what was chosen and leaves the sheets on disk; it does not score.

Measured local-vs-paid agreement
--------------------------------

``scripts/floor-vlm-agreement.py`` replays archived ``selection.json`` files
against their contact sheets, which are still on disk, so the identical question
can be re-asked locally for **zero API spend**. Over the 90 archived scenes that
still have all six sheets, where ``gpt-5.6`` chose planes 0-5 (a constant answer
would score 27.8%):

===================  ==========  ==========  ============  ===========
profile              agreement   peak VRAM   per question  weight load
===================  ==========  ==========  ============  ===========
``qwen3_vl_2b``          45.6%    5099 MiB         2.5 s        ~10 s
``qwen3_vl_4b``          61.1%    9892 MiB         2.9 s        ~15 s
``qwen3_vl_8b``          77.8%   18308 MiB         2.5 s        ~28 s
``qwen3_vl_32b`` NF4     92.2%   22022 MiB         5.3 s       ~116 s
===================  ==========  ==========  ============  ===========

Agreement rises monotonically with size, and the small models fail in one
diagnosable way: they collapse onto **plane 0**, which is the top-ranked
*structural* candidate. The 2B answered 0 for 61 of 90 scenes and the 4B for 50,
against 18 for the paid model -- so a cheap profile does not select a floor so much
as ratify the geometry ranking this arm exists to second-guess.

**That table cannot be extended, and the reason matters.** Those 90 scenes have
since been pruned to the 8 of the 2026-08-06 Replica validation, and those 8 were
built on ``qwen3_vl_8b`` rather than the paid model -- so a replay today compares
a profile against *the 8B's* choices, not ``gpt-5.6``'s, on a tenth of the sample.
Nothing measured after 2026-08-09 belongs in the rows above. Re-archive a paid
run before claiming any new profile beats the 32B.

Measured on those 8 surviving scenes on 2026-08-09, reference ``qwen3_vl_8b``,
where the best constant answer scores 50%:

=====================  ==========  ==========  ============  ==========
profile                agreement   peak VRAM   per question  bare JSON
=====================  ==========  ==========  ============  ==========
``qwen3_5_35b_a3b``      6/8 (75%)  36371 MiB       39.0 s        0/8
=====================  ==========  ==========  ============  ==========

**Why the default is ``qwen3_vl_32b``, and why the bigger model lost.** The 32B
was always the accuracy winner in the first table; what kept ``qwen3_vl_8b`` in
front of it was load time under the old ``local`` backend, and ``served`` removed
that. The 35B was then tried against it as the largest bundled profile, and the
head-to-head above is the result: **the two agree with each other on 8 of 8**,
including both scenes where they part from the archived 8B choice, where they
pick the same alternative. At n=8 against a 50% baseline neither is separated
from the 8B, and neither is separated from the other -- but agreeing 8/8 with a
profile that costs a seventh as much is not an argument for paying more.

What *is* well measured is the cost: 39.0 s per question against 5.3 s (a
256-expert MoE walks its hit experts one matmul at a time, and one selection is a
batch of one -- the worst case for that loop), 36297 MiB of VRAM against 20999,
and 0 of 8 answers arriving as bare JSON where the 32B managed 8 of 8 here and 90
of 90 on the older archive.

That last one is the sharp edge, and the deciding one. The 35B narrates before it
answers, so every selection reaches ``_parse_json_object``'s regex fallback rather
than ``json.loads``. Under ``vlm_required: true`` that regex would be the only
thing between a prose preamble and a failed build.

**It does not work at all without ``enable_thinking=False``.** Qwen3.5 chat
templates end the prompt with a bare ``<think>``, so the 1000-token budget is
spent reasoning and the answer never arrives: the same 8 scenes scored 1/8 with
**7 replies containing no JSON whatsoever**, at a 97.2 s median. That flag is now
passed by ``vlm.captioning_api.generate_conversations``, which is the one
primitive both the served and in-process paths share; see its docstring. Every
Qwen3.5 profile has this template and no Qwen3-VL or SmolVLM2 one does.

Under the old ``local`` backend a big profile was unaffordable for a second
reason: the weight load was paid *per step* on a card the step held throughout,
so ~150 floor steps serialized into hours. On ``served`` residency pays it once
for the whole campaign, which is what makes any of these affordable at all.

Running the 35B anyway is one override, plus raising the
``service.resources.gpu_memory_mib`` to 40960 in
``configs/floor_alignment/_floor_vlm_base.yaml`` -- this stage's own line now,
not a figure shared with whatever else runs the same checkpoint::

    PlaneDetectionConfig.vlm_local_profile=qwen3_5_35b_a3b

Read as caveats, not fine print: ``gpt-5.6`` is not ground truth, only the previous
behaviour, so this is *agreement* and not correctness; it is first-pass selection,
before verification; and at n=90 the binomial standard error is ~4.5 points near
78% and ~2.8 near 92%, so the ordering in the first table is solid but small gaps
are not. At n=8 the standard error is ~15 points and no gap is.

See ``docs/local-vlm.md`` for the vendored suite itself and
``notes/floor-vlm-agreement-*.json`` for the per-scene reports.
"""
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any, Callable, Sequence
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from traj_align.utils.jax_batched_map import DEFAULT_BATCH_SIZE as _JAX_BATCH_SIZE, map_batches
from traj_align.utils.sampling import uniform_subsample_indices
from traj_align.atomic_json import write_json_atomic
from traj_align.common.geometry import (
    _quaternion_to_matrix_jax,
    plane_distances_jax,
)
from .logging import LOGGER, _report_progress
from .types import (
    FloorAlignment, VlmFloorSelectionError, _ObservationView,
    _PlaneContactSheet, _PlaneVisualSample, _VlmFloorChoice, _VlmFloorElimination,
    _VlmFloorResult,
)
from traj_align.common.types import PlaneDetectionConfig, Pose, PlaneSummary
from traj_align.vlm.client import (
    VlmClient,
    VlmImage,
    VlmRequest,
    VlmUnavailableError,
    vlm_provenance,
)


def _sample_host_points(key: jax.Array, points: np.ndarray, limit: int) -> np.ndarray:
    """Sample image payload points with JAX before returning to the host.

    Looks like ``geometry._sample_points_jax`` (same random, index-sorted
    subsample idea), but not a duplicate to fold together. ``points`` here is
    a host-resident ``np.ndarray`` -- this module's callers stay on the host
    for image projection and PIL work apart from this one random draw -- so
    only the small index array is round-tripped through JAX, then pulled
    back with ``jax.device_get`` before indexing the host array. Routing
    through ``_sample_points_jax`` instead would mean copying the whole
    (potentially large) point payload to the device and back for every call,
    which is exactly the round trip this function exists to avoid.
    """

    if len(points) <= limit:
        return points
    indices = jax.random.choice(key, len(points), (limit,), replace=False)
    return points[np.asarray(jax.device_get(jnp.sort(indices)))]

def _vlm_model_label(config: PlaneDetectionConfig) -> str:
    """What to call "the model" in logs and the audit, for the active backend.

    A local run records the profile (``qwen3_vl_8b``), not ``config.vlm_model``,
    which only ever names the OpenAI checkpoint. Recording the wrong one would
    make a campaign's arms look comparable to the pre-migration ``gpt-5.6``
    numbers they are not comparable to.
    """

    return vlm_provenance(
        config.vlm_backend, model=config.vlm_model, local_profile=config.vlm_local_profile,
        served_model=config.vlm_served_model,
    )


def _refine_floor_with_vlm(
    key: jax.Array,
    points: np.ndarray,
    alignment: FloorAlignment,
    *,
    observations: Path,
    output_dir: Path,
    dataset: Path | None,
    camera: dict[str, float] | None,
    config: PlaneDetectionConfig,
    client: VlmClient,
    progress: Callable[[str], None] | None = None,
) -> FloorAlignment:

    DEFAULT_CHOICE = alignment.with_selected_floor(alignment.floor_index, method=0)

    def degrade(reason: str) -> FloorAlignment:
        """Fall back to the geometry floor, recording *why* the VLM did not choose.

        The reason has to leave this function: without it the run reports the
        ``floor_vlm_*`` arm it selected and silently publishes the geometry arm's
        number. ``FloorSelection.vlm_error`` carries it into
        ``floor_alignment.json``, where ``stages.floor.floor_alignment_status``
        turns it into the unit's status message and a dashboard event.
        """

        if config.vlm_required:
            raise VlmFloorSelectionError(
                f"VLM floor selection was configured but did not happen: {reason} "
                "Set PlaneDetectionConfig.vlm_required=false to accept the geometry "
                "floor, or vlm_enabled=false to select the geometry arm outright."
            )
        LOGGER.warning("[floor-vlm] %s Using geometry floor selection.", reason)
        return DEFAULT_CHOICE.with_vlm_error(reason)

    _report_progress(progress, "Loading posed observations for VLM floor selection.")
    LOGGER.info("[floor-vlm] Loading posed observations from %s.", observations)
    views = _load_observation_views(observations, dataset)
    if not views:
        return degrade("No usable posed observation images were available.")

    artifacts_dir = output_dir / "floor_plane_vlm"
    LOGGER.info(
        "[floor-vlm] Found %d usable posed images (%d with depth); creating VLM contact sheets in %s.",
        len(views), sum(view.depth is not None for view in views), artifacts_dir,
    )
    candidate_planes = alignment.planes.take(max(1, config.vlm_candidate_planes))
    _report_progress(progress, f"Generating VLM contact sheets for {len(candidate_planes)} plane candidates.")
    LOGGER.info(
        "[floor-vlm] Limiting VLM evidence to the top %d structural candidates (of %d total planes): indices=%s.",
        len(candidate_planes), len(alignment.planes), list(range(len(candidate_planes))),
    )
    if len(candidate_planes) < len(alignment.planes):
        # Logged at the boundary because it is the one bound here that can make
        # every downstream answer wrong while every downstream answer still looks
        # reasoned: the plane the model names is always one it was shown. Both
        # scenes this arm failed on in the 2026-08-09 campaign failed this way.
        LOGGER.info(
            "[floor-vlm] Planes %s were detected but get no contact sheet and cannot be selected; "
            "raise PlaneDetectionConfig.vlm_candidate_planes to include them.",
            list(range(len(candidate_planes), len(alignment.planes))),
        )
    key, rng = jax.random.split(key)
    sheets = _write_plane_contact_sheets(
        rng,
        points,
        candidate_planes,
        views,
        output_dir=artifacts_dir,
        camera=camera,
        config=config,
        progress=progress,
    )
    if not sheets:
        return degrade("Could not create any candidate contact sheets.")

    audit_path = artifacts_dir / "selection.json"
    rounds = max(1, int(config.vlm_selection_rounds))
    multi_turn = rounds > 1
    turns: list[dict[str, Any]] = []

    def write_audit(result: _VlmFloorResult | None = None) -> None:
        """Rewrite the audit as it stands, per turn.

        Always `sheets`, never the live ballot: `candidates[]` is the record of what
        was *rendered*, which `scripts/floor-vlm-agreement.py` replays against and a
        human uses to look at a plane the scheme voted out. What the ballot was on
        each turn is in `selection_turns` instead.
        """

        _write_vlm_audit(
            audit_path,
            observations=observations,
            dataset=dataset,
            views=views,
            planes=alignment.planes,
            sheets=sheets,
            config=config,
            result=result,
            turns=turns if multi_turn else None,
        )

    write_audit()
    model_label = _vlm_model_label(config)
    LOGGER.info(
        "[floor-vlm] Sending %d candidate contact sheets to %s. Audit manifest: %s",
        len(sheets),
        model_label,
        audit_path,
    )
    live = list(sheets)
    drop = _elimination_count(len(sheets))
    if multi_turn:
        LOGGER.info(
            "[floor-vlm] Selecting by elimination: %d turns, voting out %d of %d candidates "
            "(%d%%) per turn before the final choice.",
            rounds, drop, len(sheets), _ELIMINATION_PERCENT,
        )

    for turn_number in range(1, rounds):
        # Clamped so the final turn is always a choice rather than a formality. The
        # clamp, not the percentage, is what makes small plane sets safe: at two
        # candidates there is nothing to eliminate, and the scheme becomes the
        # single-turn question it is being compared against.
        drop_this = min(drop, max(0, len(live) - _MINIMUM_FINAL_BALLOT))
        if drop_this <= 0:
            reason = (
                f"only {len(live)} candidates remain; at least {_MINIMUM_FINAL_BALLOT} must reach "
                "the final turn"
            )
            LOGGER.info("[floor-vlm] Elimination turn %d/%d skipped: %s.", turn_number, rounds, reason)
            _report_progress(progress, f"VLM selection turn {turn_number}/{rounds}: skipped ({reason}).")
            turns.append({
                "turn": turn_number,
                "kind": "elimination",
                "prompt_style": config.vlm_prompt_style,
                "live_plane_indices": [sheet.plane_index for sheet in live],
                "skipped": reason,
            })
            write_audit()
            continue
        _report_progress(
            progress,
            f"VLM selection turn {turn_number}/{rounds}: asking {model_label} to vote "
            f"{drop_this} of {len(live)} candidates out.",
        )
        elimination = _ask_vlm_to_eliminate_floors(alignment.planes, live, drop_this, config, client)
        turns.append({
            "turn": turn_number,
            "kind": "elimination",
            "prompt_style": config.vlm_prompt_style,
            "live_plane_indices": [sheet.plane_index for sheet in live],
            "contact_sheets": [str(sheet.path) for sheet in live],
            "requested_eliminations": drop_this,
            "eliminated_plane_indices": list(elimination.eliminated),
            "answered_plane_indices": list(elimination.answered),
            "discarded_votes": elimination.discarded,
            "skipped": None,
            "attempts": list(elimination.attempts),
            "parsed": elimination.parsed,
            "raw_response": elimination.response,
            "error": elimination.error,
            "response_diagnostics": elimination.response_diagnostics,
        })
        # Recorded before it is acted on, and written before the next turn asks
        # anything, so a step killed mid-selection leaves the reason on disk.
        write_audit()
        if elimination.error:
            # **Never a degrade.** An elimination turn is an optimization of the
            # ballot, not the decision: keeping every candidate costs a wasted call
            # and leaves the final turn exactly the question a single-turn arm asks.
            # Failing the build here would triple the number of calls that can kill
            # one, for no gain over falling back to the wider ballot.
            LOGGER.warning(
                "[floor-vlm] Elimination turn %d/%d eliminated nothing (%s); the ballot stays at %d.",
                turn_number, rounds, elimination.error, len(live),
            )
            _report_progress(
                progress,
                f"VLM selection turn {turn_number}/{rounds}: no usable vote; keeping all "
                f"{len(live)} candidates.",
            )
            continue
        dropped = set(elimination.eliminated)
        live = [sheet for sheet in live if sheet.plane_index not in dropped]
        LOGGER.info(
            "[floor-vlm] Elimination turn %d/%d voted out planes %s; %d remain: %s.",
            turn_number, rounds, sorted(dropped), len(live),
            [sheet.plane_index for sheet in live],
        )
        _report_progress(
            progress,
            f"VLM selection turn {turn_number}/{rounds}: eliminated {sorted(dropped)}; "
            f"{len(live)} candidates remain.",
        )

    _report_progress(progress, f"VLM selection turn {rounds}/{rounds}: asking {model_label} to choose from {len(live)} contact sheets.")
    vlm_result = _ask_vlm_for_floor(alignment.planes, live, config, client)
    _report_progress(progress, f"VLM selection turn {rounds}/{rounds}: response received; recording the decision.")
    if multi_turn:
        ballot = [sheet.plane_index for sheet in live]
        chosen = vlm_result.choice.plane_index if vlm_result.choice else None
        on_ballot = chosen in ballot if chosen is not None else None
        if on_ballot is False:
            # `_ask_vlm_for_floor` validates the index against every detected plane,
            # not against the ballot, and that check is deliberately left alone --
            # tightening it would fail builds that succeed today. So this is the one
            # way the scheme can be entirely ineffective while every record reads
            # confident, and it says so here rather than nowhere.
            LOGGER.warning(
                "[floor-vlm] The final turn chose plane %d, which was not on the ballot %s; "
                "the elimination turns had no effect on this answer.",
                chosen, ballot,
            )
        turns.append({
            "turn": rounds,
            "kind": "selection",
            "prompt_style": config.vlm_prompt_style,
            "live_plane_indices": ballot,
            "contact_sheets": [str(sheet.path) for sheet in live],
            "selected_plane_index": chosen,
            "on_ballot": on_ballot,
            "parsed": vlm_result.choice.parsed if vlm_result.choice else vlm_result.parsed,
            "raw_response": vlm_result.choice.response if vlm_result.choice else vlm_result.response,
            "error": vlm_result.error,
            "response_diagnostics": vlm_result.response_diagnostics,
            # Same key, same shape as every elimination turn's, so a reader walking
            # `selection_turns` sees the repair rounds of the decisive turn the same
            # way it sees the others rather than having to know the last one is special.
            "attempts": list(vlm_result.attempts),
        })
    narrowed_ballot_had_no_answer = (
        vlm_result.choice is None
        and _OUT_OF_RANGE_ERROR in (vlm_result.error or "")
    )
    if narrowed_ballot_had_no_answer and multi_turn and len(live) < len(sheets):
        # **Re-ask the undiminished question before failing the build.**
        #
        # The elimination turns narrow the ballot, and when they vote the real
        # floor off it the decisive turn has no right answer left to give. What
        # the model actually does then is answer out of range -- observed as a
        # flat `plane_index: -1`, three times running, on arm a04 of the
        # 2026-09-14 sweep. Each attempt inside `_ask_vlm_for_floor` builds a
        # *fresh* prompt rather than extending a conversation, so re-asking the
        # same narrowed ballot reproduces the same answer; the arm rebuilt
        # cleanly only because a second mapping run detected different planes.
        # That is luck, not recovery, and `vlm_required: true` turns it into a
        # lost arm the rest of the time. Every affected arm in that sweep --
        # a02, a04, a07, a12, r01 -- runs an elimination profile, and no
        # single-turn arm has ever produced this answer.
        #
        # So this is the rule the elimination-error branch above already states,
        # applied one step later: *an elimination turn is an optimization of the
        # ballot, not the decision*. There it falls back to the wider ballot when
        # a turn errors; here it does the same when the narrowed ballot yields no
        # usable answer. The retry asks exactly what a single-turn arm asks, so
        # the worst case is one extra call and the behaviour of the arm this one
        # is measured against.
        #
        # Deliberately not a loop and deliberately not a relaxation of
        # `vlm_required`: if the model cannot name a floor among *every* detected
        # plane, it has said something real about the scene and the build should
        # fail with that on the record.
        #
        # Keyed on `_OUT_OF_RANGE_ERROR` and not merely on "no choice", because
        # only this failure says anything about the *ballot*. A reply that is
        # empty, prose, or JSON without an integer index says the model misread
        # the answer *shape*, which a wider ballot cannot help -- re-asking there
        # would double the calls that can kill a build for no gain, and
        # `test_only_the_final_turn_can_fail_the_build` fails when it does.
        # The raw answer goes to the *step log*, not only to `selection.json`.
        # `stages.setup` rmtree's the scene's map output directory before a
        # rebuild, so the audit recording a failure is deleted by the next run on
        # that trajectory -- which is exactly the run someone makes when trying to
        # understand the failure. On 2026-09-14 that destroyed the only record of
        # a04's `plane_index: -1`, leaving the cause of this fallback unproven:
        # "elimination voted the real floor off the ballot" and "the model
        # refuses under `survey` with this many sheets" predict the same symptom
        # and want opposite fixes. The step log outlives the build directory, so
        # the next occurrence settles it.
        LOGGER.warning(
            "[floor-vlm] The narrowed ballot %s yielded no usable answer (%s); "
            "re-asking over all %d detected candidates. Raw answers were: %s",
            [sheet.plane_index for sheet in live],
            vlm_result.error,
            len(sheets),
            [str(a.get("raw_response"))[:300] for a in vlm_result.attempts],
        )
        _report_progress(
            progress,
            f"VLM selection: no usable answer from {len(live)} candidates; "
            f"re-asking over all {len(sheets)}.",
        )
        rewidened = _ask_vlm_for_floor(alignment.planes, sheets, config, client)
        turns.append({
            "turn": rounds,
            "kind": "selection-rewidened",
            "prompt_style": config.vlm_prompt_style,
            "live_plane_indices": [sheet.plane_index for sheet in sheets],
            "contact_sheets": [str(sheet.path) for sheet in sheets],
            "selected_plane_index": (
                rewidened.choice.plane_index if rewidened.choice else None
            ),
            "on_ballot": True,
            "narrowed_ballot": [sheet.plane_index for sheet in live],
            "narrowed_error": vlm_result.error,
            "parsed": rewidened.choice.parsed if rewidened.choice else rewidened.parsed,
            "raw_response": (
                rewidened.choice.response if rewidened.choice else rewidened.response
            ),
            "error": rewidened.error,
            "response_diagnostics": rewidened.response_diagnostics,
            "attempts": list(rewidened.attempts),
        })
        vlm_result = rewidened

    write_audit(vlm_result)
    if vlm_result.choice is None:
        # The audit above is already on disk with the raw response and
        # diagnostics, so `degrade` may raise from here without losing them.
        #
        # ...on disk, but not for long: `stages.setup` rmtree's the scene's map
        # output directory at the start of a rebuild, so the audit explaining a
        # failure is deleted by the next run on that trajectory -- which is the
        # run someone makes *because* of the failure. Mirror the raw answers into
        # the step log, which outlives the build directory, so a lost arm can be
        # diagnosed from the logs alone rather than only from an artifact that
        # the act of retrying destroys.
        LOGGER.warning(
            "[floor-vlm] Giving up on VLM floor selection after %d turn(s) over "
            "%d detected candidates (%s). Raw answers from the decisive turn "
            "were: %s",
            len(turns),
            len(sheets),
            vlm_result.error,
            [str(attempt.get("raw_response"))[:300] for attempt in vlm_result.attempts],
        )
        return degrade(f"VLM floor selection failed: {vlm_result.error}")

    choice = vlm_result.choice
    LOGGER.info(
        "[floor-vlm] VLM selected plane %d as floor (model=%s, parsed=%s).",
        choice.plane_index,
        model_label,
        json.dumps(choice.parsed, sort_keys=True),
    )
    REFINED_CHOICE = alignment.with_selected_floor(choice.plane_index, method=1)
    return REFINED_CHOICE



def _load_observation_views(observations: Path, dataset: Path | None) -> list[_ObservationView]:
    views: list[_ObservationView] = []
    with observations.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            payload = json.loads(raw_line)
            image_value = payload.get("image")
            pose_value = payload.get("pose")
            if not image_value or not isinstance(pose_value, dict):
                continue
            image = _resolve_observation_image(str(image_value), observations.parent, dataset)
            if not image.is_file():
                continue
            try:
                pose = Pose.from_dict({name: float(pose_value[name]) for name in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")})
            except (KeyError, TypeError, ValueError):
                continue
            depth_value = payload.get("depth")
            depth = _resolve_observation_image(str(depth_value), observations.parent, dataset) if depth_value else None
            views.append(_ObservationView(index=index, image=image, depth=depth if depth and depth.is_file() else None, pose=pose))
    return views


def _resolve_observation_image(value: str, observation_dir: Path, dataset: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    if dataset is not None:
        candidate = dataset.expanduser().resolve() / path
        if candidate.exists():
            return candidate
    return observation_dir / path


def _write_plane_contact_sheets(
    key: jax.Array,
    points: np.ndarray,
    planes: tuple[PlaneSummary, ...],
    views: list[_ObservationView],
    *,
    output_dir: Path,
    camera: dict[str, float] | None,
    config: PlaneDetectionConfig,
    progress: Callable[[str], None] | None = None,
) -> list[_PlaneContactSheet]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_views = _uniform_sample(views, config.vlm_max_observations)
    sheets: list[_PlaneContactSheet] = []
    for plane_index, plane in enumerate(planes):
        key, rng = jax.random.split(key)
        samples = _representative_plane_samples(rng, points, plane, sampled_views, camera, config)
        if not samples:
            LOGGER.warning("[floor-vlm] Plane %d has no usable representative images.", plane_index)
            continue
        sheet = output_dir / f"plane_{plane_index:02d}.jpg"
        _write_contact_sheet(sheet, plane_index, plane, samples, config.vlm_images_per_plane)
        sheets.append(_PlaneContactSheet(plane_index, sheet, tuple(samples)))
        _report_progress(progress, f"Saved VLM contact sheet {len(sheets)}/{len(planes)} for plane {plane_index}.")
        LOGGER.info("[floor-vlm] Candidate plane %d contact sheet: %s", plane_index, sheet)
        for sample in samples:
            LOGGER.info(
                "[floor-vlm]   plane %d uses observation %d: %s (depth=%s score=%.1f, visible_projected_points=%d).",
                plane_index,
                sample.observation_index,
                sample.image,
                next((view.depth for view in views if view.index == sample.observation_index), None),
                sample.score,
                len(sample.pixels),
            )
    return sheets
def _representative_plane_samples(
    key: jax.Array,
    points: np.ndarray,
    plane: PlaneSummary,
    views: list[_ObservationView],
    camera: dict[str, float] | None,
    config: PlaneDetectionConfig,
) -> list[_PlaneVisualSample]:
    distances = map_batches(
        plane_distances_jax,
        points,
        plane.normal,
        plane.offset,
        in_axes=(0, None, None),
        batch_size=_JAX_BATCH_SIZE,
    )
    plane_points = points[np.asarray(jax.device_get(distances <= config.distance_tolerance))]
    if len(plane_points) == 0:
        return []
    key, rng = jax.random.split(key)
    plane_points = _sample_host_points(rng, plane_points, config.vlm_max_projected_points)

    scored: list[_PlaneVisualSample] = []
    for view in views:
        size = _image_size(view.image)
        if size is None:
            continue
        key, rng = jax.random.split(key)
        score, pixels = _score_projected_plane(rng, plane_points, view.pose, size, camera, depth=view.depth)
        if score > 0:
            scored.append(_PlaneVisualSample(view.index, view.image, score, pixels))

    if not scored:
        return [
            _PlaneVisualSample(view.index, view.image, 0.0, ())
            for view in _uniform_sample(views, config.vlm_images_per_plane)
        ]
    return _select_diverse_samples(scored, config.vlm_images_per_plane)


def _score_projected_plane(
    key: jax.Array,
    points: np.ndarray,
    pose: Pose | dict[str, float],
    size: tuple[int, int],
    camera: dict[str, float] | None,
    *,
    depth: Path | None,
) -> tuple[float, tuple[tuple[float, float], ...]]:
    """Project world points using the documented camera-to-world observation pose.

    The exporter writes ``Twc``. With row-major NumPy points, world-to-camera
    is therefore ``(world - camera_position) @ Rwc``. Trying the transpose as
    a second hypothesis makes behind-camera and occluded planes look visible.
    """
    pose = Pose.from_dict(pose) if isinstance(pose, dict) else pose
    orientation = _quaternion_to_matrix_jax(pose.rotation_xyzw)
    camera_points = np.asarray(jax.device_get(map_batches(
        _world_to_camera_points_jax,
        points,
        pose.translation,
        orientation,
        in_axes=(0, None, None),
        batch_size=_JAX_BATCH_SIZE,
    )))
    # Projection is a visualization boundary. JAX commonly runs this kernel
    # in float32; remove sub-micrometre quaternion round-off before converting
    # the points into image coordinates on the host.
    camera_points = np.round(camera_points, decimals=6)
    pixels = _project_camera_points(key, camera_points, size, camera, depth=depth)
    if not pixels:
        return 0.0, ()
    xs = np.asarray([pixel[0] for pixel in pixels], dtype=float)
    ys = np.asarray([pixel[1] for pixel in pixels], dtype=float)
    width, height = size
    coverage = ((float(xs.max() - xs.min()) / max(width, 1)) + (float(ys.max() - ys.min()) / max(height, 1))) / 2
    return len(pixels) + 100.0 * coverage, pixels


@jax.jit
def _world_to_camera_points_jax(
    points: jax.Array, translation: jax.Array, world_to_camera_rotation: jax.Array
) -> jax.Array:
    return (points - translation) @ world_to_camera_rotation


def _project_camera_points(
    key: jax.Array,
    points: np.ndarray,
    size: tuple[int, int],
    camera: dict[str, float] | None,
    *,
    depth: Path | None,
) -> tuple[tuple[float, float], ...]:
    width, height = size
    fx = float((camera or {}).get("fx", 600.0))
    fy = float((camera or {}).get("fy", 600.0))
    cx = float((camera or {}).get("cx", width / 2.0))
    cy = float((camera or {}).get("cy", height / 2.0))
    z = points[:, 2]
    valid = z > 0.05
    if not bool(valid.any()):
        return ()
    visible = points[valid]
    u = fx * visible[:, 0] / visible[:, 2] + cx
    v = fy * visible[:, 1] / visible[:, 2] + cy
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not bool(in_frame.any()):
        return ()
    visible = visible[in_frame]
    pixels = np.column_stack([u[in_frame], v[in_frame]])
    if depth is not None:
        depth_image = _load_depth_image(depth)
        if depth_image is not None:
            px = np.clip(np.rint(pixels[:, 0]).astype(int), 0, depth_image.shape[1] - 1)
            py = np.clip(np.rint(pixels[:, 1]).astype(int), 0, depth_image.shape[0] - 1)
            observed_depth = depth_image[py, px] / float((camera or {}).get("depth_scale", 1.0))
            # Permit a little registration/slam error, but reject map points
            # hidden behind the surface seen by this RGB-D frame.
            depth_matches = (observed_depth <= 0) | (visible[:, 2] <= observed_depth + 0.10)
            pixels = pixels[depth_matches]
    if len(pixels) > 350:
        pixels = _sample_host_points(key, pixels, 350)
    return tuple((float(x), float(y)) for x, y in pixels)


def _load_depth_image(path: Path) -> np.ndarray | None:
    try:
        with Image.open(path) as image:
            return np.asarray(image, dtype=float)
    except OSError:
        LOGGER.warning("[floor-vlm] Could not load depth image for contact-sheet visibility filtering: %s", path)
        return None


def _select_diverse_samples(samples: list[_PlaneVisualSample], count: int) -> list[_PlaneVisualSample]:
    """Return the highest-scoring visual samples.

    The sample payloads live on the host, but their numeric scores can be
    ranked in one vectorized top-k operation.
    """
    limit = min(max(count, 0), len(samples))
    if limit == 0:
        return []
    scores = jnp.asarray([sample.score for sample in samples])
    _, indices = jax.lax.top_k(scores, limit)
    return [samples[int(index)] for index in indices]


def _write_contact_sheet(
    destination: Path,
    plane_index: int,
    plane: PlaneSummary,
    samples: list[_PlaneVisualSample],
    images_per_plane: int,
) -> None:
    panel_width = 560
    label_height = 42
    panels: list[Image.Image] = []
    for sample_index, sample in enumerate(samples[:images_per_plane], start=1):
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
        scale = min(panel_width / max(image.width, 1), 1.0)
        panel_height = max(1, int(image.height * scale))
        image = image.resize((int(image.width * scale), panel_height))
        panel = Image.new("RGB", (panel_width, panel_height + label_height), "white")
        panel.paste(image, ((panel_width - image.width) // 2, label_height))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, panel_width, label_height), fill=(255, 255, 255))
        label = f"Plane {plane_index} / view {sample_index} / score {sample.score:.1f}"
        draw.text((12, 12), label, fill=(20, 20, 20), font=ImageFont.load_default())
        for x, y in sample.pixels:
            px = (panel_width - image.width) // 2 + int(x * scale)
            py = label_height + int(y * scale)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 45, 45), outline=(255, 255, 255))
        panels.append(panel)

    if not panels:
        return
    metadata_height = 56
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels) + metadata_height
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    normal = ", ".join(f"{value:.3f}" for value in plane.normal)
    draw.text(
        (12, 14),
        f"Plane {plane_index}: normal=({normal}), support={plane.support_ratio:.3f}, extent=({plane.extent[0]:.2f}, {plane.extent[1]:.2f})",
        fill=(20, 20, 20),
        font=ImageFont.load_default(),
    )
    x_offset = 0
    for panel in panels:
        sheet.paste(panel, (x_offset, metadata_height))
        x_offset += panel.width
    sheet.save(destination, "JPEG", quality=84, optimize=True)


def _floor_candidate_line(index: int, plane: PlaneSummary) -> str:
    return (
        f"Plane {index}: normal={tuple(round(value, 4) for value in plane.normal)}, "
        f"support_ratio={plane.support_ratio:.4f}, extent={tuple(round(value, 3) for value in plane.extent)}, "
        f"structural_score={plane.structural_score:.4f}"
    )


# **Do not reintroduce an up direction derived from the camera poses here.** It was
# tried on 2026-08-10 and removed the same day: each candidate line carried a
# `tilt_from_upright` computed by averaging each pose's image-up axis
# (``R @ (0, -1, 0)``), on the reasoning that a walking capture is upright, and it
# did change office1's answer from a wall to a plausible floor.
#
# The reasoning is wrong for this mapper, and wrong in the way that matters. ORB-
# SLAM3's poses are self-consistent along a trajectory but carry no relationship to
# the world frame -- their gauge is arbitrary, which is the entire reason this stage
# exists. So "camera up" is up in a frame nobody has fixed yet, and a plane's angle
# against it is a number with no physical meaning, delivered to the model as though
# it had one. It happened to look right on Replica; there is no reason it would
# stay right, and a wrong tilt is worse than no tilt because the prompt told the
# model to eliminate candidates by it.
#
# The same objection applies to any measure of "which plane is the floor" taken
# from pose orientation -- including scoring this stage's accuracy that way, which
# is why no such script ships. Evidence about a floor selection has to come from
# the contact sheets, or from something in the map that does not need gravity
# (a plane with a third of the cloud beneath it is not the floor, whatever the
# frame).


def floor_selection_prompt(
    candidate_lines: Sequence[str],
    style: str = "direct",
    repair: str | None = None,
) -> str:
    """The floor-selection question, from the candidates' numeric summaries.

    Extracted (not reworded) so ``scripts/floor-vlm-agreement.py`` can replay
    archived selections against *this* prompt instead of a hand-copied duplicate:
    the replay's agreement number is only evidence about the local backend if it
    asks the question the pipeline actually asks.

    **Do not tune the ``direct`` wording without re-running that replay.** The
    archived ``selection.json`` files record what ``gpt-5.6`` answered to *those
    words*, so changing them costs the only free comparison available -- and the
    obvious change has already been measured and rejected:

    Replacing the literal ``{"plane_index": 0, "confidence": 0.82, ...}`` example
    with placeholders looked well-motivated. Under the literal example the local
    8B returned ``confidence: 0.82`` on all 90 archived scenes and chose plane 0
    for 33 of them (the paid model chose 0 for 18), which reads exactly like
    example-copying. It is not: with placeholders the plane-0 count only moved
    33 -> 30, agreement went 77.8% -> 74.4% (inside the ~4.5% binomial standard
    error at n=90, so no real change), and the model started fencing its JSON in
    ```json blocks, leaning on ``_parse_json_object``'s regex on every single
    scene instead of none. The plane-0 lean is the model favouring the top
    structural candidate, and an added "a high structural_score is not by itself a
    reason" sentence did not shift it either.

    That is why the second wording is a *new style* beside it rather than an edit
    to it. ``survey`` asks for the same decision but makes the answer's JSON name
    what each plane's red points sit on before it names the winner. Under greedy
    decoding the survey is in the context window when ``plane_index`` is emitted,
    so it is reasoning-before-answering with three properties thinking mode does
    not have: it costs tens of tokens rather than thousands, it needs no
    hybrid-reasoning template (so the Qwen3-VL profiles get it too), and it lands
    in ``selection.json`` where an operator auditing a suspicious floor can read
    why the model said what it said. ``vlm_thinking_enabled`` remains the way to
    get a Qwen3.5 profile's *native* reasoning; the two are independent.

    It is also where the *un-truncated* reasoning goes. ``qwen3_5_35b_a3b`` under
    ``direct`` narrates each candidate in prose before answering and does not
    reach the JSON: on 4 offline trials over identical sheets, 2 answers were cut
    off mid-sentence and the other 2 named a wall. It was already surveying; it
    just had nowhere to put the survey and no budget to finish it.

    Both styles end with an explicit "only these indices are eligible" clause, and
    that clause is worth more than either: with ``vlm_candidate_planes: 12`` the
    numeric summaries and the sheets finally cover the same set, so a model naming
    a plane it was told about is naming a plane it was shown.

    ``repair`` is appended last, exactly as in ``floor_elimination_prompt`` and for
    exactly the same reason: decoding is greedy, so re-sending an identical prompt
    returns identical bytes and a retry has to say what was wrong with the previous
    reply. **It does not touch the freeze above.** ``repair`` is ``None`` on the
    first attempt, which is the only attempt any archived ``selection.json`` ever
    recorded, so ``scripts/floor-vlm-agreement.py`` still replays against byte-identical
    words. It is also the reason the clauses live in ``_SELECTION_REPAIRS`` beside
    this function rather than being folded into the styles: a style is a question
    the archive can be replayed against, a repair is a correction that only a
    live conversation ever sees.
    """

    if style not in _PROMPT_STYLES:
        raise ValueError(
            f"Unknown floor prompt style {style!r}; expected one of {sorted(_PROMPT_STYLES)}"
        )
    shared = (
        "You are selecting the true floor plane from a 3D indoor map. "
        "Each attached image is a contact sheet for one candidate plane; red points mark visible projected 3D map points belonging to that candidate. "
        "Choose the candidate that corresponds to the walkable floor, not a wall, ceiling, table, counter, bed, or other object. "
        "A candidate is a floor only when the red points trace the large walkable surface across multiple views; reject a candidate whose points sit on a wall, furniture, screen, or ceiling. "
        "Only the attached plane indices are eligible. Use the images first and the numeric metadata as supporting evidence.\n\n"
        + "\n".join(candidate_lines)
    )
    prompt = shared + _PROMPT_STYLES[style]
    if repair:
        prompt += f"\n\n{repair}"
    return prompt


#: The tail of each prompt style; the question above it is identical.
#:
#: ``survey`` adds a warning about the normals as well as the answer shape, and it
#: is not cosmetic. The candidate lines report ``normal`` in the *mapper's* frame,
#: which has no gravity direction in it, and models read it as though it did:
#: ``qwen3_5_35b_a3b`` justified a wall with "the large negative X-component
#: indicates a plane facing..." and a wall-with-upward-normal with "negative Y,
#: consistent with a floor" in the same run. The Open3D verification prompt in
#: ``traj_align.floor.open3d`` has always carried this warning ("must never
#: be used to judge rendered level"); the selection prompt did not, and the
#: mistake it invites is the exact one this stage exists to prevent. It goes in
#: this style only because ``direct`` is the wording the archive can be replayed
#: against.
_PROMPT_STYLES = {
    "direct": '\n\nReturn only JSON like {"plane_index": 0, "confidence": 0.82, "reason": "short reason"}',
    "survey": (
        "\n\nThe normal above is expressed in the mapper's own arbitrary frame, which does not "
        "encode gravity: no component of it tells you which direction is up, so never infer "
        "floor, wall or ceiling from the normal's sign or largest component. Two candidates are "
        "parallel when their normals match up to sign, and that is all these numbers can tell you. "
        "Judge orientation only from the images.\n\n"
        "First survey every attached candidate, then choose. Return only JSON like "
        '{"survey": {"0": "red points on a table top", "1": "red points on the walkable carpet"}, '
        '"plane_index": 1, "confidence": 0.82, "reason": "short reason"}, '
        "with one survey entry per attached plane index, each at most eight words naming the "
        "surface its red points actually lie on."
    ),
}


#: How many times the *decisive* question is asked before the build fails. Total
#: attempts, not retries: 3 is one question and two corrections.
#:
#: This is the answer ``vlm_required`` fails a build over, and until 2026-08-12 it
#: was asked exactly once. That cost a build: on the 20-job campaign that day,
#: ``room2`` on the single-turn 35B arm answered with prose and no JSON object, and
#: the step had nowhere to go but ``degrade``. ``qwen3_5_35b_a3b`` was already
#: rejected as the default partly on JSON robustness (see
#: ``configs/floor_alignment/_floor_vlm_base.yaml``), so an arm that runs it is
#: an arm that meets this, and a model that misreads the answer *shape* has not said
#: anything about the floor -- unlike a model that names the wrong plane, which this
#: cannot help with and must not try to.
#:
#: Matched to ``_ELIMINATION_MAX_ATTEMPTS`` deliberately. They are separate constants
#: because they bound separate questions with different stakes (an elimination turn
#: that gives up keeps the ballot; this one fails the build), but nothing has yet
#: argued for different numbers, and two different numbers would be a claim.
_SELECTION_MAX_ATTEMPTS = 3

#: The one selection failure that says something about the *ballot* rather than
#: about the model's grasp of the answer shape. Named once because two places
#: depend on the spelling: `_read_selection_answer` writes it, and the
#: re-widening fallback in `_select_floor_with_vlm` keys off it.
_OUT_OF_RANGE_ERROR = "VLM returned out-of-range plane_index"

#: What each unusable selection answer is told, keyed by what was wrong with it.
#:
#: Every clause names the failure and restates the answer shape, because a model
#: that produced no ``plane_index`` may have misread either. They are literals here
#: rather than f-strings at the point of failure so that the whole vocabulary of
#: corrections this stage can send is readable in one place -- the same reason
#: ``_PROMPT_STYLES`` is a table.
_SELECTION_REPAIRS = {
    "empty": (
        "Your previous reply was empty. Reply with nothing but the JSON object described "
        'above, for example {"plane_index": 0, "confidence": 0.82, "reason": "short reason"}.'
    ),
    "not_json": (
        "Your previous reply did not contain the required JSON object. Do not explain your "
        "reasoning in prose. Reply with nothing but the JSON object described above, for "
        'example {"plane_index": 0, "confidence": 0.82, "reason": "short reason"}.'
    ),
    "no_index": (
        'Your previous reply was JSON but had no integer "plane_index". The answer must '
        'contain "plane_index" set to one of the attached plane indices as a bare integer, '
        'not a string and not a list, for example {"plane_index": 0, "confidence": 0.82, '
        '"reason": "short reason"}.'
    ),
}


def _out_of_range_repair(plane_index: int, plane_count: int) -> str:
    """The one repair clause that has to quote the answer back.

    Spelling out the legal range is the whole correction here, and unlike the other
    three failures this one has a number the model chose -- naming it is what tells
    a model that invented an index from a model that misnumbered a real candidate.
    """

    return (
        f"Your previous reply chose plane_index {plane_index}, which does not exist. "
        f"Choose one of the attached plane indices; they are integers from 0 to "
        f"{plane_count - 1}. Reply with nothing but the JSON object described above."
    )


#: Share of the *original* ballot each elimination turn votes out, as an integer
#: percent. Not a config field on purpose: it is the scheme's definition rather
#: than a per-arm knob, and the prompt states it as a count of planes rather than
#: as a percentage (a model asked for "30%" has to do the arithmetic itself, and
#: the arithmetic is the one part of this question a program should own). Recorded
#: into ``selection.json``'s ``selection_scheme``, which is the only place it is
#: written down at run time.
_ELIMINATION_PERCENT = 30

#: How many times one elimination turn is asked before it gives up and keeps the
#: whole ballot. Giving up is not a build failure -- see
#: ``PlaneDetectionConfig.vlm_selection_rounds``.
_ELIMINATION_MAX_ATTEMPTS = 3

#: Never let an elimination turn narrow the ballot below this. A final turn with
#: one candidate is not a choice, and the scheme should degrade to a no-op on a
#: tiny plane set rather than to a forced answer.
_MINIMUM_FINAL_BALLOT = 2


def _elimination_count(candidate_count: int) -> int:
    """How many candidates each elimination turn votes out.

    Integer half-up on purpose, rather than ``round(0.30 * candidate_count)``:
    ``0.3 * 15`` is 4.4999... in float64, so a float rounding would return 4 where
    every human reading "30% of 15" expects 5 -- and ``round`` is banker's anyway,
    so it disagrees with itself on the halves. This arithmetic ends up as a literal
    integer in a prompt, so it is worth having no way to be surprised by it.

    Derived from the *original* ballot and reused unchanged for every elimination
    turn, which is what makes the scheme legible: at 12 candidates it is always 4,
    so the turns run 12 -> 8 -> 4 rather than 12 -> 8 -> 6.
    """

    return max(1, (_ELIMINATION_PERCENT * candidate_count + 50) // 100)


def floor_elimination_prompt(
    candidate_lines: Sequence[str],
    live_plane_indices: Sequence[int],
    drop_count: int,
    style: str = "direct",
    repair: str | None = None,
) -> str:
    """The elimination question: which candidates leave the ballot this turn.

    A different question from ``floor_selection_prompt`` and worth keeping as one,
    rather than a mode of it. The selection prompt asks for the best candidate and
    is answered with ``plane_index``; this one asks for the worst ``drop_count`` and
    is answered with ``eliminate``. Distinct keys mean a malformed elimination reply
    can never be read as a selection, which matters because these answers narrow the
    ballot silently.

    Three things this wording does that the selection wording does not have to:

    * **The count is a literal integer**, twice -- in the instruction and beside the
      answer shape. The caller owns the arithmetic (``_elimination_count``); the
      model is only asked to count planes.
    * **The ballot is listed explicitly.** After turn 1 the live set is a sparse
      subset of ``0..N-1`` and "the attached ones" stops being a usable description
      of it. The sheets carry their original plane index in their own label, and so
      does every candidate line, so the indices never renumber between turns.
    * **It invites naming fewer.** A model told to eliminate exactly four of eight
      will find a fourth, and the fourth can be the floor. An elimination is
      irreversible -- no later turn can put a plane back -- so the prompt and
      ``_sanitise_elimination_votes`` agree that fewer is a valid answer.

    ``repair`` is appended last and is what makes a retry worth making at all:
    decoding is greedy (``captioning_api.generate_conversations`` passes
    ``do_sample=False``, and ``vlm.service`` *rejects* ``temperature`` rather than
    ignoring it), so re-sending an identical prompt returns identical bytes. A retry
    therefore has to say what was wrong with the previous reply.

    Unlike ``floor_selection_prompt``'s ``direct`` style, nothing here is frozen: no
    archived selection ever answered these words, so ``scripts/floor-vlm-agreement.py``
    has nothing to replay against them and this text may be tuned freely. The one
    thing to preserve is that **both** styles carry the warning about the normals:
    this is the turn where reading a normal's sign as gravity ("negative Y,
    consistent with a floor" -- ``qwen3_5_35b_a3b``, on a wall) takes the floor off
    the ballot for good.
    """

    if style not in _ELIMINATION_STYLES:
        raise ValueError(
            f"Unknown floor prompt style {style!r}; expected one of {sorted(_ELIMINATION_STYLES)}"
        )
    ballot = ", ".join(str(index) for index in live_plane_indices)
    shared = (
        "You are narrowing down the candidates for the true floor plane of a 3D indoor map. "
        "Each attached image is a contact sheet for one candidate plane; red points mark visible projected 3D map points belonging to that candidate. "
        f"Vote to eliminate exactly {drop_count} of these {len(live_plane_indices)} candidates: the {drop_count} least likely to be the walkable floor. "
        "A candidate is a floor only when the red points trace the large walkable surface across multiple views; a candidate whose points sit on a wall, furniture, screen, or ceiling is one to eliminate. "
        f"Eliminate only candidates you are confident are not the floor: if fewer than {drop_count} are clearly not the floor, name fewer. "
        "A candidate you eliminate is gone for good and cannot win later, so never eliminate a plane that might be the walkable floor. "
        f"Only these plane indices are on the ballot: {ballot}. Each attached contact sheet is labelled with its own plane index; answer with those numbers. "
        "Use the images first and the numeric metadata as supporting evidence.\n\n"
        + "\n".join(candidate_lines)
        + "\n\nThe normal above is expressed in the mapper's own arbitrary frame, which does not "
        "encode gravity: no component of it tells you which direction is up, so never infer "
        "floor, wall or ceiling from the normal's sign or largest component. Two candidates are "
        "parallel when their normals match up to sign, and that is all these numbers can tell you. "
        "Judge orientation only from the images."
    )
    prompt = shared + _ELIMINATION_STYLES[style].format(drop_count=drop_count)
    if repair:
        prompt += f"\n\n{repair}"
    return prompt


#: The tail of each elimination style. Keyed identically to ``_PROMPT_STYLES``,
#: because ``vlm_prompt_style`` is one setting for the whole selection: a style
#: present there and missing here would raise mid-build on an arm that has already
#: paid for its contact sheets.
#:
#: The example lists are literal rather than placeholders, for the reason recorded
#: on ``floor_selection_prompt``: placeholders measurably pushed models into fencing
#: their JSON in ``` blocks, which puts every answer through
#: ``_parse_json_object``'s regex fallback. The literal example shows two indices
#: whatever ``drop_count`` is, so the count is restated next to it.
_ELIMINATION_STYLES = {
    "direct": (
        '\n\nReturn only JSON like {{"eliminate": [1, 4], "reason": "short reason"}}, '
        'where "eliminate" holds exactly {drop_count} plane indices -- the ones you are voting '
        "out -- or fewer if fewer are clearly not the floor."
    ),
    "survey": (
        "\n\nFirst survey every attached candidate, then vote. Return only JSON like "
        '{{"survey": {{"0": "red points on a table top", "1": "red points on the walkable carpet"}}, '
        '"eliminate": [0], "reason": "short reason"}}, '
        "with one survey entry per attached plane index, each at most eight words naming the "
        'surface its red points actually lie on, and "eliminate" holding exactly {drop_count} '
        "plane indices -- or fewer if fewer are clearly not the floor."
    ),
}


def _complete_vlm_request(
    client: VlmClient, request: VlmRequest
) -> tuple[str, dict[str, Any], str | None, bool]:
    """Ask one floor question. Returns ``(raw, diagnostics, error, retriable)``.

    Both floor questions go through here, and that is the point: the exception
    triage below is the only thing standing between a lost service and a build that
    silently publishes the geometry arm under the VLM arm's label, and a second
    hand-written copy of it is exactly how that guarantee gets lost.

    ``retriable`` is true only for the one failure a *different prompt* could fix --
    the backend answered with no text. A backend that cannot be asked at all, or a
    call that failed, will not answer differently for being asked again.
    """

    try:
        response = client.complete(request)
    except VlmUnavailableError as exc:
        # Not a failed attempt -- the backend could not be asked at all (no API
        # key, no SDK, no such local profile, weights that will not fit). Recorded
        # the same way so `degrade` reports it identically either way.
        return "", {}, str(exc), False
    except Exception as exc:  # noqa: BLE001 - preserve the reason in alignment metadata.
        # For the remote backend this is reached only once the retry budget is
        # spent, so a `RateLimitError` recorded here means sustained pressure
        # rather than one unlucky call. The client describes its own failures:
        # formatting a local processor error as an OpenAI status would be a lie in
        # the one field a human reads to find out what happened.
        error, diagnostics = client.describe_failure(exc)
        return "", diagnostics, error, False
    diagnostics = response.diagnostics
    raw = response.text
    LOGGER.info("[floor-vlm] Raw VLM response: %s", raw)
    if not raw:
        return "", diagnostics, _empty_vlm_response_message(diagnostics), True
    return raw, diagnostics, None, False


def _sanitise_elimination_votes(
    parsed: dict[str, Any] | None, live: Sequence[int], drop_count: int
) -> tuple[tuple[int, ...], tuple[int, ...], str | None, str | None]:
    """Read one elimination answer. Returns ``(applied, answered, discarded, repair)``.

    ``applied`` is what actually leaves the ballot: on the live set, de-duplicated,
    in the model's own order, and never more than the turn asked for. ``answered``
    is every integer the model named, including the ones thrown away, because the
    difference between the two is the only way an audit reader can tell a model that
    understood the ballot from one that guessed at it.

    ``repair`` is non-``None`` when the answer was not usable and a corrective clause
    should be added to the next attempt's prompt. The distinction that matters:

    * an ``eliminate`` list the model deliberately left **empty** is a usable answer
      ("nothing here is clearly not the floor"), because the prompt invites it, so it
      eliminates nothing and is not retried;
    * an answer with no usable list at all, or one where every named index is off the
      ballot, is a misunderstanding, and is retried with the misunderstanding named.

    Nothing here can fail a build -- the worst outcome is a turn that eliminates
    nothing and a ballot that stays as wide as it was.
    """

    example = '{"eliminate": [1, 4], "reason": "short reason"}'
    if not parsed or "eliminate" not in parsed:
        return (), (), None, (
            "Your previous reply did not contain the required JSON object. Reply with "
            f"nothing but the JSON, exactly like {example}."
        )
    raw_votes = parsed["eliminate"]
    if isinstance(raw_votes, (int, float)) and not isinstance(raw_votes, bool):
        raw_votes = [raw_votes]
    if not isinstance(raw_votes, (list, tuple)):
        return (), (), None, (
            f'Your previous reply\'s "eliminate" was not a list of plane indices. Reply with '
            f"nothing but the JSON, exactly like {example}."
        )
    answered: list[int] = []
    for value in raw_votes:
        if isinstance(value, bool):
            continue
        try:
            answered.append(int(value))
        except (TypeError, ValueError):
            continue
    live_set = set(live)
    off_ballot = [index for index in answered if index not in live_set]
    applied: list[int] = []
    duplicates: list[int] = []
    for index in answered:
        if index not in live_set:
            continue
        if index in applied:
            duplicates.append(index)
            continue
        applied.append(index)
    surplus = applied[drop_count:]
    applied = applied[:drop_count]

    notes = []
    if off_ballot:
        notes.append(f"not on the ballot: {off_ballot}")
    if duplicates:
        notes.append(f"named more than once: {duplicates}")
    if surplus:
        notes.append(f"beyond the {drop_count} requested: {surplus}")
    discarded = "; ".join(notes) or None

    if answered and not applied:
        ballot = ", ".join(str(index) for index in live)
        return (), tuple(answered), discarded, (
            f"Your previous reply named {off_ballot}, and none of those are on the ballot. "
            f"The eligible plane indices are {ballot}, and they are the numbers printed on the "
            "attached contact sheets."
        )
    return tuple(applied), tuple(answered), discarded, None


def _ask_vlm_to_eliminate_floors(
    planes: tuple[PlaneSummary, ...],
    sheets: list[_PlaneContactSheet],
    drop_count: int,
    config: PlaneDetectionConfig,
    client: VlmClient,
) -> _VlmFloorElimination:
    """Run one elimination turn, re-asking a misunderstood question up to twice.

    The retries are the reason this is a loop rather than a call. They are bounded
    by ``_ELIMINATION_MAX_ATTEMPTS``, they only happen for answers a corrected prompt
    could plausibly fix (``_sanitise_elimination_votes`` decides, and a transport
    failure is never one), and every attempt keeps its own record -- the reply, what
    was wrong with it, and the clause the next prompt carried -- so "the repair
    clause is doing the work on this scene" is something an operator can read rather
    than infer.
    """

    live = [sheet.plane_index for sheet in sheets]
    candidate_lines = [
        _floor_candidate_line(index, plane)
        for index, plane in enumerate(planes)
        if index in set(live)
    ]
    attempts: list[dict[str, Any]] = []
    repair: str | None = None
    for attempt_number in range(1, _ELIMINATION_MAX_ATTEMPTS + 1):
        prompt = floor_elimination_prompt(
            candidate_lines, live, drop_count, config.vlm_prompt_style, repair
        )
        request = VlmRequest(
            prompt=prompt,
            images=tuple(VlmImage(sheet.path) for sheet in sheets),
            max_tokens=config.vlm_answer_max_tokens,
            enable_thinking=config.vlm_thinking_enabled,
            purpose="floor elimination",
        )
        # Logged *before* the call, not after it. Everything else about a floor
        # question is recorded once an answer (or a returned error) exists, which
        # left the one outcome that produces neither -- a raised `GpuServiceLost`
        # from a service that went away mid-step -- as a log that simply stopped, with the
        # last line being "Selecting by elimination" and no way to tell a hung call
        # from a dead one. This is the line that says a question was actually asked.
        LOGGER.info(
            "[floor-vlm] Elimination attempt %d/%d: asking %s to vote %d of %d out; "
            "ballot=%s prompt_chars=%d images=%d max_tokens=%d repair=%s.",
            attempt_number, _ELIMINATION_MAX_ATTEMPTS, _vlm_model_label(config), drop_count,
            len(live), live, len(prompt), len(request.images),
            config.vlm_answer_max_tokens, repair is not None,
        )
        raw, diagnostics, error, retriable = _complete_vlm_request(client, request)
        parsed = _parse_json_object(raw) if raw else None
        applied, answered, discarded, next_repair = (
            ((), (), None, None) if error else _sanitise_elimination_votes(parsed, live, drop_count)
        )
        attempts.append({
            "attempt": attempt_number,
            "repair_clause": repair,
            "raw_response": raw,
            "parsed": parsed,
            "error": error,
            "response_diagnostics": diagnostics,
        })
        if error and not retriable:
            return _VlmFloorElimination(
                attempts=tuple(attempts), error=error, response_diagnostics=diagnostics
            )
        if error:
            repair = (
                "Your previous reply was empty. Reply with nothing but the JSON object "
                'described above, for example {"eliminate": [1, 4], "reason": "short reason"}.'
            )
            continue
        if next_repair is None:
            return _VlmFloorElimination(
                eliminated=applied,
                answered=answered,
                attempts=tuple(attempts),
                discarded=discarded,
                parsed=parsed or {},
                response=raw,
                response_diagnostics=diagnostics,
            )
        LOGGER.warning(
            "[floor-vlm] Elimination attempt %d/%d was not usable; re-asking with a correction.",
            attempt_number, _ELIMINATION_MAX_ATTEMPTS,
        )
        repair = next_repair
    last = attempts[-1]
    return _VlmFloorElimination(
        answered=(),
        attempts=tuple(attempts),
        error=(
            f"No usable elimination vote after {_ELIMINATION_MAX_ATTEMPTS} attempts "
            f"(last reply: {(last['raw_response'] or '')[:200]!r})."
        ),
        response=last["raw_response"] or "",
        response_diagnostics=last["response_diagnostics"],
    )


def _read_selection_answer(
    parsed: dict[str, Any] | None, plane_count: int
) -> tuple[int | None, str | None, str | None]:
    """Read one selection answer. Returns ``(plane_index, error, repair)``.

    The counterpart of ``_sanitise_elimination_votes``, and split out of
    ``_ask_vlm_for_floor`` for the same reason: the loop should not be the only
    place that knows which answers are worth re-asking.

    ``error`` and ``repair`` are both ``None`` exactly when the answer is usable,
    and both non-``None`` otherwise -- **every** content failure here is repairable,
    which is the substantive difference from the elimination reader. All three are
    the model misreading the *answer shape*, and a corrected prompt can plausibly
    fix all three. The failures a repair cannot fix never reach this function:
    ``_complete_vlm_request`` triages them, and a transport error is never handed
    here.

    The out-of-range check stays against every *detected* plane rather than against
    the elimination ballot, unchanged and deliberately so -- see the caller in
    ``_select_floor_with_vlm``, where naming an eliminated plane is recorded and not
    fatal. Tightening it here would fail builds that succeed today.
    """

    if not parsed:
        return None, "VLM response did not contain a JSON object.", _SELECTION_REPAIRS["not_json"]
    try:
        plane_index = int(parsed["plane_index"])
    except (KeyError, TypeError, ValueError):
        return (
            None,
            "VLM response did not include an integer plane_index.",
            _SELECTION_REPAIRS["no_index"],
        )
    if plane_index < 0 or plane_index >= plane_count:
        return (
            None,
            f"{_OUT_OF_RANGE_ERROR} {plane_index}.",
            _out_of_range_repair(plane_index, plane_count),
        )
    return plane_index, None, None


def _ask_vlm_for_floor(
    planes: tuple[PlaneSummary, ...],
    sheets: list[_PlaneContactSheet],
    config: PlaneDetectionConfig,
    client: VlmClient,
) -> _VlmFloorResult:
    """Ask the decisive floor question, re-asking a misread one up to twice.

    A loop rather than a call for the same reason ``_ask_vlm_to_eliminate_floors``
    is, with one difference in the stakes that is worth stating: an elimination turn
    that exhausts its budget keeps the ballot and the build continues, while this
    one exhausting its budget is a failed build under ``vlm_required``. That is the
    argument for the loop, not against it.

    What it does **not** do is retry a *decision*. The budget is spent only on
    answers ``_read_selection_answer`` says a corrected prompt could fix -- a reply
    with no JSON, no integer ``plane_index``, or an index that does not exist -- plus
    the one transport outcome ``_complete_vlm_request`` marks retriable, an answer
    with no text at all. A model that names a wrong-but-real plane is answering the
    question, gets its answer taken, and is not second-guessed here: re-asking until
    a different plane comes back would be sampling for the answer this stage prefers,
    and under greedy decoding it would not even do that -- it would just spend the
    step's time returning the same index.
    """

    candidate_indices = {sheet.plane_index for sheet in sheets}
    candidate_lines = [
        _floor_candidate_line(index, plane)
        for index, plane in enumerate(planes)
        if index in candidate_indices
    ]
    images = tuple(VlmImage(sheet.path) for sheet in sheets)
    attempts: list[dict[str, Any]] = []
    repair: str | None = None
    for attempt_number in range(1, _SELECTION_MAX_ATTEMPTS + 1):
        prompt = floor_selection_prompt(candidate_lines, config.vlm_prompt_style, repair)
        request = VlmRequest(
            prompt=prompt,
            images=images,
            max_tokens=config.vlm_answer_max_tokens,
            enable_thinking=config.vlm_thinking_enabled,
            purpose="floor selection",
        )
        # Logged before the call, as in `_ask_vlm_to_eliminate_floors` and for the
        # same reason: a `GpuServiceLost` raised out of `client.complete` produces
        # neither an answer nor a recorded error, so without this line a floor log
        # simply stops and a hung call reads exactly like a dead one.
        LOGGER.info(
            "[floor-vlm] Selection attempt %d/%d: asking %s to choose 1 of %d candidates; "
            "ballot=%s prompt_chars=%d images=%d max_tokens=%d repair=%s.",
            attempt_number, _SELECTION_MAX_ATTEMPTS, _vlm_model_label(config),
            len(candidate_lines), sorted(candidate_indices), len(prompt), len(images),
            config.vlm_answer_max_tokens, repair is not None,
        )
        raw, diagnostics, error, retriable = _complete_vlm_request(client, request)
        parsed = _parse_json_object(raw) if raw else None
        plane_index, answer_error, next_repair = (
            (None, error, None) if error else _read_selection_answer(parsed, len(planes))
        )
        attempts.append({
            "attempt": attempt_number,
            "repair_clause": repair,
            "raw_response": raw,
            "parsed": parsed,
            "error": error or answer_error,
            "response_diagnostics": diagnostics,
        })
        if error and not retriable:
            # No key, no SDK, a call that failed post-backoff. Not a misunderstanding,
            # so re-asking spends the step's remaining time to fail the same way.
            return _VlmFloorResult(
                None,
                error,
                response=raw,
                parsed=parsed or {},
                response_diagnostics=diagnostics,
                attempts=tuple(attempts),
            )
        if error:
            repair = _SELECTION_REPAIRS["empty"]
            continue
        if plane_index is not None:
            return _VlmFloorResult(
                _VlmFloorChoice(
                    plane_index=plane_index,
                    response=raw,
                    parsed=parsed or {},
                    artifacts=tuple(str(sheet.path) for sheet in sheets),
                ),
                response=raw,
                parsed=parsed or {},
                response_diagnostics=diagnostics,
                attempts=tuple(attempts),
            )
        LOGGER.warning(
            "[floor-vlm] Selection attempt %d/%d was not usable (%s); re-asking with a correction.",
            attempt_number, _SELECTION_MAX_ATTEMPTS, answer_error,
        )
        repair = next_repair
    last = attempts[-1]
    # The base message is kept verbatim at the front. It is what lands in
    # `floor_alignment.json`'s `vlm_error` and on the dashboard, and readers --
    # including `tests/test_floor.py` -- match on it; the suffix adds the fact that
    # the budget was spent rather than replacing what was wrong.
    return _VlmFloorResult(
        None,
        f"{last['error']} (no usable answer after {_SELECTION_MAX_ATTEMPTS} attempts)",
        response=last["raw_response"] or "",
        parsed=last["parsed"] or {},
        response_diagnostics=last["response_diagnostics"],
        attempts=tuple(attempts),
    )


def _write_vlm_audit(
    destination: Path,
    *,
    observations: Path,
    dataset: Path | None,
    views: list[_ObservationView],
    planes: tuple[PlaneSummary, ...],
    sheets: list[_PlaneContactSheet],
    config: PlaneDetectionConfig,
    result: _VlmFloorResult | None = None,
    turns: Sequence[dict[str, Any]] | None = None,
) -> None:
    """Write a human-readable trace of every image and decision sent to the VLM.

    ``turns`` is the elimination scheme's record and is omitted entirely on a
    single-turn run, so this file keeps the exact shape every existing reader was
    written against (``scripts/floor-vlm-agreement.py``, ``scripts/floor-vlm-trial.py``).
    ``vlm_result`` still holds the winner whatever the scheme, for the same reason:
    the archive-replay contract is that one key.
    """

    sheet_by_plane = {sheet.plane_index: sheet for sheet in sheets}
    payload: dict[str, Any] = {
        "observations": str(observations),
        "dataset": str(dataset) if dataset else None,
        "backend": config.vlm_backend,
        "model": _vlm_model_label(config),
        "usable_observation_count": len(views),
        "usable_observations": [
            {"observation_index": view.index, "image": str(view.image), "depth": str(view.depth) if view.depth else None, "pose": view.pose.to_dict()}
            for view in views
        ],
        "candidates": [],
    }
    for plane_index, plane in enumerate(planes):
        sheet = sheet_by_plane.get(plane_index)
        payload["candidates"].append(plane.to_dict() | {
            "plane_index": plane_index,
            "contact_sheet": str(sheet.path) if sheet else None,
            "selected_source_images": [
                {
                    "observation_index": sample.observation_index,
                    "image": str(sample.image),
                    "projection_score": sample.score,
                    "projected_point_count": len(sample.pixels),
                }
                for sample in sheet.samples
            ]
            if sheet
            else [],
        })
    if turns:
        payload["selection_scheme"] = {
            "rounds": max(1, int(config.vlm_selection_rounds)),
            "candidate_count": len(sheets),
            # Both constants live in this module rather than in the config, so this
            # is the only place a finished build records what "30%" and "3 attempts"
            # meant for it. A reader comparing two arms needs them.
            "elimination_percent": _ELIMINATION_PERCENT,
            "eliminated_per_turn": _elimination_count(len(sheets)),
            "attempts_per_turn": _ELIMINATION_MAX_ATTEMPTS,
            "minimum_final_ballot": _MINIMUM_FINAL_BALLOT,
        }
        payload["selection_turns"] = list(turns)
    if result is not None:
        payload["vlm_result"] = {
            "selected_plane_index": result.choice.plane_index if result.choice else None,
            "parsed": result.choice.parsed if result.choice else result.parsed,
            "raw_response": result.choice.response if result.choice else result.response,
            "error": result.error,
            "response_diagnostics": result.response_diagnostics,
            # Both arms record these, not just the multi-turn one: `selection_scheme`
            # below is written only when there were elimination turns, and the repair
            # budget applies to the single-turn arm just as much. `attempts` is one
            # entry on a clean answer, so its length is how a reader tells a first-try
            # answer from a repaired one without parsing the error string.
            "attempts": list(result.attempts),
            "max_attempts": _SELECTION_MAX_ATTEMPTS,
        }
    write_json_atomic(destination, payload)
    LOGGER.info("[floor-vlm] Wrote VLM selection audit: %s", destination)


#: How to name a backend that answered with nothing, per
#: ``VlmResponse.diagnostics["backend"]``. Kept explicit rather than formatting
#: the raw key, because this string lands in ``floor_alignment.json``'s
#: ``vlm_error`` and on the dashboard.
_BACKEND_LABELS = {"openai": "OpenAI Responses API", "local": "The local VLM"}


def _empty_vlm_response_message(diagnostics: dict[str, Any]) -> str:
    """Explain an empty result without discarding what the backend did report.

    The Responses API can say *why* it produced no text
    (``incomplete_details.reason``, ``error.code``); a local ``generate`` call
    cannot, and reports only the model it ran. Both go through here, so whichever
    fields exist are the ones named -- rather than asserting a shape only one
    backend has.
    """

    backend = str(diagnostics.get("backend") or "openai")
    details = [f"status={diagnostics.get('status') or 'unknown'}"]
    for key in ("incomplete_reason", "error_code", "error_message", "response_id", "model"):
        value = diagnostics.get(key)
        if value:
            details.append(f"{key}={value}")
    output_types = diagnostics.get("output_types")
    if output_types:
        details.append(f"output_types={output_types}")
    label = _BACKEND_LABELS.get(backend, backend)
    return f"{label} returned no text output (" + ", ".join(details) + ")."


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, dict) else None


def _uniform_sample(items: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    indices = uniform_subsample_indices(len(items), limit)
    return [items[int(index)] for index in indices]


def _image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except OSError:
        return None
