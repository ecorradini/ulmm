# Access redundancy κ — code and data

Reproduction package for *Route-Diversity Indices Cannot Certify Access Redundancy:
Theory and a Five-City Audit* (IEEE TNSE submission; supersedes earlier drafts titled
"…An Exact Max-Flow Resilience Measure and the Misranking of Route-Diversity Indices"
and "…What Route-Diversity Indices Cannot Certify").

The access redundancy `κ(d)` of a demand `d` is the maximum number of edge-disjoint directed
paths from `d` to an access set on a friction-aware street substrate; by Menger's theorem it
equals the minimum number of substrate edges whose removal isolates `d` (the exact
targeted-attack budget), and its minimum cut names the responsible streets.

## Layout

```
code/      analysis scripts (Python; run from the repository root)
data/      zipped extracts (unzip in place before running — see Setup)
results/   per-demand result files behind every table and figure (CSV, ready to inspect)
```

## Data availability

The five street substrates (OSMnx, network type `drive_service`), the 500 m demand grids, and
the last-mile access inventories (OpenStreetMap `amenity=post_office|parcel_locker`; retail and
service POI tags as in Appendix B of the paper) were extracted in **September 2025**. The
independent New York City extract used in the discriminant-validity test was downloaded in
**October 2025**, and the healthcare access sets (`amenity=hospital|clinic|doctors`) in
**June 2026**. Population weights use the **WorldPop 2020 1 km UN-adjusted** rasters; the crash
and 311 records are **2024 NYC Open Data** exports.

| `data/` archive | Unzips to | Contents |
| --- | --- | --- |
| `ulmm_pickles.zip` | `ulmm_pickles/` | the 5 city ULMM substrates + the 3 European healthcare variants + size summaries |
| `cache_nyc_ulmm.zip` | `cache_nyc_ulmm/` | the independent NYC extract (`graph_G`, `segments_fe`, anchors, demand weights) and the 2024 truck-crash / 311 record exports |
| `cache_pop.zip` | `cache_pop/` | the WorldPop 2020 1 km rasters (NLD, ESP, FRA, USA) |

The Seattle and New York City healthcare access sets are queried live from OpenStreetMap by
`kappa_healthcare.py` (no fixed snapshot); the three European healthcare substrates are shipped
as cached ULMM pickles built by `build_healthcare.py`.

## Setup

```bash
pip install -r requirements.txt        # Python 3.12; OSMnx pulls in GeoPandas/Shapely/Rasterio
cd <repo root>
unzip -o data/ulmm_pickles.zip         # -> ulmm_pickles/
unzip -o data/cache_nyc_ulmm.zip       # -> cache_nyc_ulmm/
unzip -o data/cache_pop.zip            # -> cache_pop/
unzip -o data/cache_nyc_derived.zip    # -> cache_nyc_ulmm/ (exact-DEB artefacts,
                                       #    lets the audits run without redoing
                                       #    the betweenness pipeline)
```

All scripts use paths relative to the repository root, so **run them from the root**, e.g.
`python code/reanchor_kappa.py`. Each writes its outputs to `results/`; the figure scripts write
PNGs to `../paper/` (edit the output path, or create a sibling `paper/` directory, for a
standalone run). A small per-process cache is created under `cache/` on first run.

## Pipeline

`ablations.py` is the shared library (`load_ulmm`, `collapsed_graph`, the κ max-flow machinery);
`build_ulmm_mod.py` / `build_healthcare.py` are the substrate builders. The result CSVs in
`results/` already reproduce every number in the paper; to regenerate them, run:

1. `reanchor_kappa.py` — primary multi-entry κ, both orientations, k∈{1,3}; cut location.
   → `results/reanchor_{perdemand,summary}.csv` (**Tables 4, 6**; timings for **Table 2**).
   The k=1 slice provides the single-entry labels used downstream (`kappa.py` recomputes the
   same single-entry κ independently as `results/kappa_perdemand.csv`).
2. `kappa_ksweep.py` — anchoring-granularity sweep k∈{1..5}+250 m (**Table 7**).
3. `misranking2.py` — diversity indices vs κ, single-entry config (**Table 5**, left).
4. `misranking_multientry.py` — same, multi-entry config (**Table 5**, right); reuses
   `misranking2.py` and `results/reanchor_perdemand.csv`.
5. `k_sensitivity.py` — edge-dissimilarity AUC vs route budget K (§ Misranking).
6. `cut_chars.py` — cut-location and entry-set diameters (§ Robustness).
7. `kappa_nodedisjoint.py` — node- vs edge-disjoint κ (§ Robustness).
8. `kappa_physical.py` — physical-segment (way-disjoint) bracket (§ Robustness, Prop. on
   antiparallel arcs).
9. `kappa_popweight.py` — population-weighted (α_pop=1) sensitivity (§ Robustness).
   `e6_robust.py` — illustration that κ is byte-identical across friction salience
   λ∈{0.5,1,2} (the invariance check cited in § Robustness).
10. `density_curve.py` — access-density sweep (**Figure 3**).
11. `cutcrit_external.py` (+ `b1_wd_external.py`) — held-out NYC discriminant-validity test
    (**Table 8**).
12. `kappa_healthcare.py` (+ `pop_weights.py`) — population-weighted healthcare instantiation
    (**Table 9**).
13. `fig_kappa.py`, `fig_spof_map.py` — **Figures 2 and 4**.
14. `toy_check.py` — end-to-end correctness assertions on synthetic networks.

κ is invariant to the cost model, so it does not depend on the friction parameters; those enter
only the route-diversity baselines (van profile).


## Revision additions (TNSE version)

New scripts (in `code/`), all runnable offline from the cached data except the closure
download:

15. `misranking_certificate.py` — the sound disjoint-subfamily certificate `D(P) ≤ κ`
    computed on the same route ensembles as the index audit (**Table IV**; acceptance
    gate: zero soundness violations across all 3,623 instances).
16. `external_spof_stats.py` — corrected NYC betweenness→SPOF retrieval statistics
    (average precision both directions, NTA stratification, within-NTA permutation,
    matched controls, zero-betweenness floor decomposition; **§V-E.1**). Persists the
    previously ad-hoc reverse block to `results/external_spof_stats.csv`.
17. `reliability_mc.py` — Monte-Carlo check of the disconnection-exponent proposition on
    Barcelona (**Table VIII**; fitted exponents cluster at κ).
18. `closure_replay.py` — predictive-validity replay of the NYC DOT "Street Closures due
    to Construction Activities by Block" full-closure permits against the October-2025
    extract (**§V-E.2**). The archived snapshot (2026-07-19) ships in
    `data/cache_closures/`; per-demand κ + named min-cut edges on the external extract
    are persisted to `results/external_perdemand_kappa.csv`.
19. `audit_numbers.py`, `audit_numbers_tnse.py` — number audits: every hand-transcribed
    manuscript value is recomputed from its source CSV (170 + 338 checks; both must exit 0).

Reporting conventions: pooled shares weight cities by **demand mass** (New York City
carries ≈44% of the pooled weight; equal-city pooling would put the pooled
single-point-of-failure share near 9% instead of 7–8%, and the manuscript states this
alternative once); co-located zones (entry node coincides with an access anchor) are reported as κ=∞ inside
the redundant class (85 zones at k=3, 5.5% of pooled weight; the code's `KCAP=50` display
cap corresponds to this convention).

## Certification-frontier additions (current TNSE version)

The theoretical core of the current version is the **certification frontier**: the least
budget `B*_r(d) = min{B : κ(G^(B)) ≥ r}` whose sub-network already carries `r` disjoint
routes is both necessary and sufficient for a budget-local generator to certify redundancy
`r` (Proposition 8). These scripts produce every number behind it.

20. `certificate_envelope.py` — completeness envelope of the certificate `D` across
    generator families and budgets (IPSP ρ∈{4,1.25}, exact Yen, ε-budgets × K∈{3,5,15};
    **Table XII**). Also persists per-demand rows with each cell's realized and a priori
    horizons to `results/certificate_envelope_perdemand.csv`.
21. `certification_frontier.py` — exact `B*_2`, `B*_3` per demand (two shortest-path
    passes + binary search over the O(m) thresholds, capped max-flow per probe) →
    `results/certification_horizon.csv`. Guard `--guard` recomputes κ on the full support
    and asserts it reproduces the stored labels.
22. `frontier_gate.py` — the falsification gate. T1 tests necessity pointwise (a single
    violation refutes Proposition 8); T2 compares each Table XII cell against its ceiling
    `P(B_g ≥ B*_2)`. Deduplicates `(city,d)` before joining: several demand cells snap to
    the same street node, and joining without that silently reweights every rate.
23. `frontier_achievability.py` — the achievability half: at `B*_2` the augmenting-path
    generator must return two arc-disjoint routes (200/200 sampled demands), plus a
    positive control comparing the binary search against an exhaustive threshold sweep.
24. `deb_exact.py` — exact demand-weighted betweenness over all 2,657 distinct
    access-anchor nodes, replacing the earlier 80-source subsample.
25. `deb_orientation_control.py` — the same betweenness recomputed in the *matching*
    demand→access orientation, the control behind §V-E.1's conclusion that no orientation
    retrieves single points of failure at usable precision.
26. `closure_replay_ext.py` — multi-street closure count and the concurrent (day-by-day
    union) replay, with severance durations and weighted-demand-days.
27. `bridge_baseline.py`, `hybrid_workflow.py`, `densify_test.py`, `pop_coverage.py`,
    `equity_overlay.py` — structural bridge baseline, screen-then-verify comparator,
    anchor densification, zero-POI population share, and the ACS tract overlay
    (keyless: streams the ACS table-based summary files).
28. `entry_diameters.py` — per-city entry-set spatial diameter by `k` (the 55–91 m span
    quoted for block scale).
29. `check_build.py` — build gates for the manuscript itself (page budget, abstract
    length, embedded fonts, no overfull boxes, references ending on page 10).

Run the audits last: `python code/audit_numbers.py` and `python code/audit_numbers_tnse.py`
must both report zero failures. On a fresh clone the five `INST` checks report `INFO`
rather than a value, because they read the collapsed-graph cache that the pipeline builds
on first run; everything else is checked against the shipped CSVs.
