# Solar Inspection local beta

The home dashboard shows saved inspections, active jobs, processed cells,
review flags, and a seven-day activity chart using real database records.
Use the collapsible sidebar to open Inspections, Reports, Activity log, or
Business profile. The responsive layout supports desktop and mobile browsers.

PDF reports are available in Reports and on completed inspection pages.
Reports include business/operator details, summary counts, a source preview,
all cell findings, and model/calibration metadata. Partial runs are clearly
marked. Set your business profile before creating new inspections; existing
reports retain the profile recorded at creation.

## Relational storage

SQLite lives at `results/web_jobs/solar_inspection.sqlite3`. Tables: `users`,
`inspections`, `items`, `findings`, and `activity_logs`. Foreign keys connect
records and cascade inspection data cleanup; activity history is retained.
Existing JSON inspections are imported automatically once. SQLite is the
authoritative store; per-job JSON remains an export/recovery mirror. Back up the
whole `results/web_jobs` folder with the server stopped to retain database and
image files together.

The current user is one shared local operator profile, not an authenticated
account. Separate accounts and remote/network sharing are not implemented.
PDF generation requires ReportLab, included in `requirements.txt`.

From the project folder, run `venv\Scripts\python.exe web_app.py`, or double-click
`start_beta.bat`. Open http://127.0.0.1:8765 and keep the terminal open.
The server uses the standard Python library and the project's existing Pillow
and ML dependencies. No additional web packages are required.

## Workflow

1. Choose individual cells or whole panels, then add images.
2. For panels, set rows, columns, independent outer margins in pixels, and cell gap. Drag a rectangle on the preview to set the boundary, or enter each margin. Pixel margins apply to every image in the batch. Check the preview
   on every image: one grid applies to the entire batch. The default is 6 × 12.
3. Optionally enable visual analysis (slower).
4. Run inspection. Jobs run sequentially; progress and per-cell errors are shown.
5. Filter findings, open a cell for probabilities and visual evidence, export CSV
   or JSON. Stop processing between cells, or delete finished inspections to
   remove their uploads and generated files.

Uploads are limited to 50 images, 20 MB each, 100 MB per batch, and 2,000 cells.
Supported formats: PNG, JPEG, BMP, TIFF. TIFF browser previews may be unavailable.
Images may be at most 20 megapixels and 10,000 pixels per side. Multi-page TIFF
uses only the first frame. Names are retained for display; internal filenames
are unique, so duplicate upload names do not overwrite one another.

Results persist in `results/web_jobs/<inspection-id>/`. JSON includes the model
hash, calibration, settings, timestamps, input names, cell locations, results,
and errors. CSV contains cell-level findings. Rejected/error results show no DSI
or confidence rather than appearing undamaged. Uploaded images are re-encoded
for safe preview without changing grayscale into artificial EL data.

## Scope and operation

This beta is **local-only** and binds to 127.0.0.1. It has no multi-user login,
remote deployment, automatic retention policy, or production security claim.
Host and origin checks restrict browser requests to the local app. Do not expose
it through a tunnel or reverse proxy. Shared deployment needs an authenticated
Python hosting environment and operational hardening.

The calibrated model loads lazily on the first inspection and stays cached.
Restart the server after replacing model/calibration files. On restart,
unfinished inspections are marked interrupted; their partial results remain
available. Restarting the server does not automatically retry them.

No training or dataset changes are performed by this interface. Predictions
require engineer review. The three excluded classes are not recognised as
unknown automatically. DSI is an indicative score, not measured power loss;
Grad-CAM/masks/skeletons are supporting visual evidence, not verified boundaries.

Tests: `venv\Scripts\python.exe -m unittest tests.test_web -v`.

After an inspection stops, use **Process rejected cells anyway** to retry only OOD-rejected cells. Inference and optional visual analysis run with the initial quality gate overridden. Results retain the original rejection reason and an OOD override flag in the UI and exports; uncertain predictions still require review. Unreadable images remain errors.
