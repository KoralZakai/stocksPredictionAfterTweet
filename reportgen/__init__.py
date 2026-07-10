"""Report generation: pure stats + pure HTML render (§13 `reporting` job).

Split so jobs/reporting.py stays a thin CLI containing zero science: it marshals
I/O and calls build_report() then render_html(). Rerun the job whenever the event
dataset or the signal cache changes and the dashboard regenerates from scratch —
no hand-edited HTML anywhere.
"""
