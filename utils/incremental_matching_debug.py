import json
from pathlib import Path

import numpy as np

from algorithms.incremental_matching import (
    IncrementalMatchingCandidate,
    IncrementalMatchingStepResult,
)
import utils.puzzle_solver_output_structure as output_structure


def build_manifest(num_puzzle_pieces: int, steps: list[dict]) -> dict:
    assert num_puzzle_pieces > 0
    return {
        "version": 1,
        "num_puzzle_pieces": int(num_puzzle_pieces),
        "num_steps": len(steps),
        "steps": steps,
    }


def step_result_to_manifest_entry(
    step_result: IncrementalMatchingStepResult,
    assets: dict[str, Path],
    incremental_matching_dir: Path,
) -> dict:
    return {
        "step": int(step_result.step),
        "accepted": _candidate_to_dict(step_result.accepted_candidate),
        "skipped_candidate_count": len(step_result.skipped_candidates),
        "skipped_candidates": [
            _candidate_to_dict(candidate)
            for candidate in step_result.skipped_candidates
        ],
        "components_before": _components_to_lists(step_result.components_before),
        "source_component_before": _component_to_list(
            step_result.source_component_before
        ),
        "target_component_before": _component_to_list(
            step_result.target_component_before
        ),
        "merged_component": _component_to_list(step_result.merged_component),
        "components_after": _components_to_lists(step_result.components_after),
        "unmatched_pieces": _component_to_list(step_result.unmatched_pieces),
        "pose_forest_edge": _component_to_list(step_result.pose_forest_edge),
        "transform": _transform_to_dict(step_result.transform),
        "assets": {
            asset_name: asset_path.relative_to(incremental_matching_dir).as_posix()
            for asset_name, asset_path in assets.items()
        },
    }


def write_debug_viewer(output_dir: Path, manifest: dict) -> None:
    manifest_path = output_structure.incremental_matching_manifest_path(output_dir)
    viewer_path = output_structure.incremental_matching_viewer_path(output_dir)
    manifest_json = json.dumps(manifest, indent=2)
    manifest_path.write_text(f"{manifest_json}\n", encoding="utf-8")
    viewer_path.write_text(
        _viewer_html(manifest_json.replace("</", "<\\/")), encoding="utf-8"
    )


def _candidate_to_dict(candidate: IncrementalMatchingCandidate) -> dict:
    return {
        "edge_index": int(candidate.edge_index),
        "source_piece": int(candidate.source_puzzle_piece_id),
        "target_piece": int(candidate.target_puzzle_piece_id),
        "source_contour": int(candidate.source_contour_index),
        "target_contour": int(candidate.target_contour_index),
        "source_match_count": int(candidate.source_match_count),
        "target_match_count": int(candidate.target_match_count),
        "match_score": int(candidate.match_score),
    }


def _components_to_lists(components: tuple[tuple[int, ...], ...]) -> list[list[int]]:
    return [_component_to_list(component) for component in components]


def _component_to_list(component: tuple[int, ...]) -> list[int]:
    return [int(puzzle_piece_id) for puzzle_piece_id in component]


def _transform_to_dict(transform) -> dict:
    return {
        "matrix": transform.params.tolist(),
        "translation": [float(value) for value in transform.translation],
        "rotation_degrees": float(np.rad2deg(transform.rotation)),
    }


def _viewer_html(manifest_json: str) -> str:
    return ("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Incremental Matching Debug Navigator</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --surface: #ffffff;
      --surface-2: #eef2f6;
      --border: #d8dee8;
      --text: #142033;
      --muted: #64748b;
      --accent: #1f6feb;
      --accent-dark: #174ea6;
      --danger: #c3312f;
      --shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    [hidden] {
      display: none !important;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
    }

    button,
    input {
      font: inherit;
    }

    .app {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }

    .sidebar {
      border-right: 1px solid var(--border);
      background: var(--surface);
      padding: 18px 14px;
      overflow-y: auto;
    }

    .brand {
      margin: 0 0 4px;
      font-size: 18px;
      line-height: 1.25;
      font-weight: 700;
    }

    .summary {
      margin: 0 0 14px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }

    .timeline {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }

    .timeline button {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 6px;
      background: transparent;
      color: var(--text);
      cursor: pointer;
      padding: 9px 10px;
      text-align: left;
    }

    .timeline button:hover {
      background: var(--surface-2);
    }

    .timeline button.active {
      border-color: rgba(31, 111, 235, 0.35);
      background: rgba(31, 111, 235, 0.08);
    }

    .timeline-step {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-size: 13px;
      font-weight: 700;
    }

    .timeline-meta {
      margin-top: 3px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
    }

    .main {
      min-width: 0;
      padding: 18px;
    }

    .toolbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: auto auto minmax(160px, 1fr) auto auto;
      gap: 10px;
      align-items: center;
      margin-bottom: 16px;
      padding: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.95);
      box-shadow: var(--shadow);
      backdrop-filter: blur(8px);
    }

    .toolbar button {
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface);
      color: var(--text);
      cursor: pointer;
      min-height: 36px;
      padding: 0 12px;
      font-size: 13px;
      font-weight: 700;
    }

    .toolbar button:hover {
      border-color: var(--accent);
      color: var(--accent-dark);
    }

    .toolbar button:disabled {
      cursor: default;
      opacity: 0.45;
    }

    .toolbar input[type="range"] {
      width: 100%;
    }

    .step-jump {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .step-jump input {
      width: 72px;
      min-height: 36px;
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0 8px;
      color: var(--text);
    }

    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 16px;
      align-items: start;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: #fbfcfe;
    }

    .panel-title {
      margin: 0;
      font-size: 14px;
      font-weight: 800;
    }

    .panel-subtitle {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .image-wrap {
      display: grid;
      min-height: 260px;
      place-items: center;
      padding: 12px;
      background: #0a0f1a;
      overflow: auto;
    }

    .image-wrap img {
      display: block;
      max-width: 100%;
      height: auto;
    }

    .details {
      display: grid;
      gap: 12px;
      padding: 14px;
    }

    .kv {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      gap: 8px;
      font-size: 13px;
      line-height: 1.45;
    }

    .kv dt {
      color: var(--muted);
      font-weight: 700;
    }

    .kv dd {
      margin: 0;
      overflow-wrap: anywhere;
    }

    .skipped {
      display: grid;
      max-height: 180px;
      overflow: auto;
      gap: 4px;
      padding: 8px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: var(--surface-2);
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }

    .empty {
      display: grid;
      min-height: 70vh;
      place-items: center;
      color: var(--muted);
      font-size: 15px;
    }

    @media (max-width: 980px) {
      .app {
        grid-template-columns: 1fr;
      }

      .sidebar {
        max-height: 220px;
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      .toolbar {
        grid-template-columns: auto auto minmax(120px, 1fr);
      }

      .step-jump {
        grid-column: 1 / -1;
      }

      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <script type="application/json" id="manifest-data">__MANIFEST_JSON__</script>
  <div class="app">
    <aside class="sidebar">
      <h1 class="brand">Incremental Matching</h1>
      <p class="summary" id="summary"></p>
      <nav class="timeline" id="timeline" aria-label="Steps"></nav>
    </aside>
    <main class="main">
      <div id="empty" class="empty" hidden>No incremental matching steps were recorded.</div>
      <section id="viewer" hidden>
        <div class="toolbar">
          <button type="button" id="prev">Prev</button>
          <button type="button" id="next">Next</button>
          <input type="range" id="slider" min="1" value="1">
          <div class="step-jump">
            <span>Step</span>
            <input type="number" id="jump" min="1" value="1">
            <span id="step-count"></span>
          </div>
          <button type="button" id="last">Last</button>
        </div>

        <div class="grid">
          <section class="panel">
            <div class="panel-header">
              <h2 class="panel-title">Matching visualization</h2>
              <span class="panel-subtitle" id="matching-caption"></span>
            </div>
            <div class="image-wrap">
              <img id="matching-image" alt="Matching visualization">
            </div>
          </section>

          <section class="panel">
            <div class="panel-header">
              <h2 class="panel-title">Pose forest</h2>
              <span class="panel-subtitle" id="pose-caption"></span>
            </div>
            <div class="image-wrap">
              <img id="pose-image" alt="Pose forest">
            </div>
          </section>

          <section class="panel">
            <div class="panel-header">
              <h2 class="panel-title">Step change</h2>
              <span class="panel-subtitle" id="change-caption"></span>
            </div>
            <div class="image-wrap">
              <img id="change-image" alt="Incremental matching change">
            </div>
          </section>

          <section class="panel">
            <div class="panel-header">
              <h2 class="panel-title">Step details</h2>
              <span class="panel-subtitle" id="details-caption"></span>
            </div>
            <div class="details">
              <dl class="kv" id="details"></dl>
              <div class="skipped" id="skipped"></div>
            </div>
          </section>
        </div>
      </section>
    </main>
  </div>

  <script>
    const manifest = JSON.parse(document.getElementById("manifest-data").textContent);
    const steps = manifest.steps || [];
    let currentIndex = 0;

    const summary = document.getElementById("summary");
    const timeline = document.getElementById("timeline");
    const viewer = document.getElementById("viewer");
    const empty = document.getElementById("empty");
    const prevButton = document.getElementById("prev");
    const nextButton = document.getElementById("next");
    const lastButton = document.getElementById("last");
    const slider = document.getElementById("slider");
    const jump = document.getElementById("jump");
    const stepCount = document.getElementById("step-count");
    const matchingImage = document.getElementById("matching-image");
    const poseImage = document.getElementById("pose-image");
    const changeImage = document.getElementById("change-image");
    const matchingCaption = document.getElementById("matching-caption");
    const poseCaption = document.getElementById("pose-caption");
    const changeCaption = document.getElementById("change-caption");
    const detailsCaption = document.getElementById("details-caption");
    const details = document.getElementById("details");
    const skipped = document.getElementById("skipped");

    summary.textContent = `${manifest.num_steps} steps, ${manifest.num_puzzle_pieces} pieces`;

    function describeCandidate(candidate) {
      return `${candidate.source_piece} -> ${candidate.target_piece}`;
    }

    function componentText(component) {
      return component.length ? component.join(", ") : "none";
    }

    function setTextRow(label, value) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      details.append(dt, dd);
    }

    function renderTimeline() {
      timeline.replaceChildren();
      steps.forEach((step, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = index === currentIndex ? "active" : "";
        const title = document.createElement("div");
        title.className = "timeline-step";
        const stepLabel = document.createElement("span");
        stepLabel.textContent = `Step ${step.step}`;
        const pairLabel = document.createElement("span");
        pairLabel.textContent = describeCandidate(step.accepted);
        title.append(stepLabel, pairLabel);

        const meta = document.createElement("div");
        meta.className = "timeline-meta";
        meta.textContent = `score ${step.accepted.match_score}, skipped ${step.skipped_candidate_count}`;

        button.append(title, meta);
        button.addEventListener("click", () => setStep(index));
        timeline.appendChild(button);
      });
    }

    function renderSkipped(step) {
      skipped.replaceChildren();
      if (!step.skipped_candidates.length) {
        skipped.textContent = "No skipped candidates before this accepted merge.";
        return;
      }
      step.skipped_candidates.forEach((candidate) => {
        const row = document.createElement("div");
        row.textContent = `edge ${candidate.edge_index}: ${describeCandidate(candidate)} score ${candidate.match_score}`;
        skipped.appendChild(row);
      });
    }

    function renderStep() {
      const step = steps[currentIndex];
      slider.value = String(currentIndex + 1);
      jump.value = String(currentIndex + 1);
      stepCount.textContent = `of ${steps.length}`;
      prevButton.disabled = currentIndex === 0;
      nextButton.disabled = currentIndex === steps.length - 1;
      lastButton.disabled = currentIndex === steps.length - 1;

      matchingImage.src = step.assets.matching_visualization;
      poseImage.src = step.assets.pose_forest;
      changeImage.src = step.assets.change_visualization;

      const accepted = step.accepted;
      matchingCaption.textContent = `step ${step.step}`;
      poseCaption.textContent = `highlight ${step.pose_forest_edge.join(" -> ")}`;
      changeCaption.textContent = `${accepted.source_piece} -> ${accepted.target_piece}`;
      detailsCaption.textContent = `score ${accepted.match_score}`;

      details.replaceChildren();
      setTextRow("Accepted pieces", `${accepted.source_piece} -> ${accepted.target_piece}`);
      setTextRow("Accepted contours", `${accepted.source_contour} -> ${accepted.target_contour}`);
      setTextRow("Match score", `${accepted.match_score} (${accepted.source_match_count} source, ${accepted.target_match_count} target)`);
      setTextRow("Transform", `rotation ${step.transform.rotation_degrees.toFixed(3)} deg, translation ${step.transform.translation.map((value) => value.toFixed(2)).join(", ")}`);
      setTextRow("Source component", componentText(step.source_component_before));
      setTextRow("Target component", componentText(step.target_component_before));
      setTextRow("Merged component", componentText(step.merged_component));
      setTextRow("Unmatched pieces", componentText(step.unmatched_pieces));
      setTextRow("Skipped before accept", String(step.skipped_candidate_count));
      renderSkipped(step);
      renderTimeline();
      preloadAdjacent();
    }

    function preloadStep(index) {
      if (index < 0 || index >= steps.length) {
        return;
      }
      Object.values(steps[index].assets).forEach((src) => {
        const image = new Image();
        image.src = src;
      });
    }

    function preloadAdjacent() {
      preloadStep(currentIndex - 1);
      preloadStep(currentIndex + 1);
    }

    function setStep(index) {
      currentIndex = Math.max(0, Math.min(steps.length - 1, index));
      renderStep();
    }

    if (!steps.length) {
      empty.hidden = false;
    } else {
      viewer.hidden = false;
      slider.max = String(steps.length);
      jump.max = String(steps.length);
      prevButton.addEventListener("click", () => setStep(currentIndex - 1));
      nextButton.addEventListener("click", () => setStep(currentIndex + 1));
      lastButton.addEventListener("click", () => setStep(steps.length - 1));
      slider.addEventListener("input", () => setStep(Number(slider.value) - 1));
      jump.addEventListener("change", () => setStep(Number(jump.value) - 1));
      document.addEventListener("keydown", (event) => {
        if (event.key === "ArrowLeft") {
          setStep(currentIndex - 1);
        } else if (event.key === "ArrowRight") {
          setStep(currentIndex + 1);
        } else if (event.key === "Home") {
          setStep(0);
        } else if (event.key === "End") {
          setStep(steps.length - 1);
        }
      });
      renderStep();
    }
  </script>
</body>
</html>
""").replace("__MANIFEST_JSON__", manifest_json)
