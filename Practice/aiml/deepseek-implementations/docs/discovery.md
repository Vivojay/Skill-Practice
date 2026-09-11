# Discovery audit — 2026-09-11

The seed is accounted for, but corpus exhaustiveness is not certified. The audit
confirms 31 arXiv identities and two distinct officially linked report files.
It adds official systems/tooling entries, not an invented additional paper count.
Non-core paper rows summarize disclosed contributions; they do not certify a
full-method reading or completed implementation of those papers.

## What was checked

| Discovery route | Observed result | Boundary |
|---|---|---|
| [Official news](https://www.deepseek.com/news/) and [English V4.1 launch](https://www.deepseek.com/en/news/deepseek-v4-1-flash/) | V4.1-Flash announcement dated 2026-09-10 links an official report | Launch language is not sufficient to implement internals |
| [Official GitHub organization](https://github.com/deepseek-ai), public repos API with per_page=100 | 40 repositories returned, below page capacity; names reviewed | Repository membership is evidence of official release, not of a distinct paper |
| [Official HF organization](https://huggingface.co/deepseek-ai) | Listing reports 105 models, including recent V4 variants, DSpark and EAGLE3 checkpoints | Full model API returned 403; four pagination attempts failed; latest visible models reviewed |
| [Open-infra index](https://github.com/deepseek-ai/open-infra-index) | Fire-Flyer and Insights reports; infrastructure project links | Hardware numbers were not reproduced |
| All 31 seed arXiv abstract pages | Canonical titles and complete visible revision histories parsed | Metadata verifies identity, not every equation or employee affiliation |
| Core full texts | MoE v1 sections 2–3; loss-free v1 method/Algorithm 1 and update ablations; V2 v5 MLA method/appendix equations; V3 v2 MTP method; Math v1 4.1 and objective appendix inspected | Precise implemented versions are fixed in notes; later revisions are not silently substituted |
| Official reference code | V3 inference MLA, rotary, gate, experts/dispatch and MIT license read; V2/MoE/Math READMEs read | HF V2/MoE modeling files returned 403; original GRPO/MTP training source not exposed by the inspected references |

Reference commits actually inspected:

| Repository | Commit | Material read |
|---|---|---|
| DeepSeek-V3 | `9b4e9788e4a3a731f7567338ed15d3ec549ce03b` | README, inference/model.py, LICENSE-CODE |
| DeepSeek-V2 | `ec98ee3cbffc32104cd55dba8af884b3d772602a` | README |
| DeepSeek-MoE | `66edeee5a4f75cbd76e0316229ad101805a90e01` | README |
| DeepSeek-Math | `b8b0f8ce093d80bf8e9a641e44142f06d092c305` | README |
| open-infra-index | `56d86855fcf6e08fdfd45ce6280bd24322c93351` | README and report links |

An extended commit/README pass hit raw-content timeouts and then GitHub's
unauthenticated API rate limit (HTTP 403). [reference-commits.json](reference-commits.json)
preserves successful commit resolution separately from unsuccessful content reads.
A resolved SHA alone is **not** labelled inspected code. Accessible browser pages
were used to read DeepJIT, DeepSelect, DeepSpec, TileKernels, deepseek-recipe,
V3.2-Exp and the relevant affiliation blocks without claiming pinned-code review.

The raw text cache is excluded at `.research/`. [source-fetch-audit.json](source-fetch-audit.json)
stores initial URLs, byte lengths, SHA256 hashes and failures;
[source-metadata.json](source-metadata.json) stores canonical identities/histories.
`python scripts/discover.py` repeats the initial network discovery (not needed for
tests or experiments). The later browser checks and searches are recorded here;
that script does not claim to reproduce an exhaustive affiliation search.

## Search log and attribution checks

Executed search strings included:

- `site:arxiv.org DeepSeek 2026 2607 2608 2609`
- `site:deepseek.com DeepSeek V4.1 September 2026`
- `site:arxiv.org "DeepSeek" "2026"`
- `site:arxiv.org "DeepSeek" affiliation new paper`
- `site:arxiv.org "DeepSeek-AI" -DeepSeekMoE -DeepSeekMath -DeepSeek-V2 -DeepSeek-V3 -DeepSeek-R1 -survey`
- `site:arxiv.org "High-Flyer" "DeepSeek" publications`
- `site:github.com/deepseek-ai "arxiv.org" "2608" OR "2609" OR "2607"`
- The EMA prior-work queries in [hypothesis.md](../hypothesis.md).

Broad searches mostly returned already-catalogued reports and third-party surveys.
For example, 2507.09955 and 2504.03219 discuss DeepSeek but were not admitted as
DeepSeek-authored publications. Employee names alone do not establish institutional
attribution. Confirmed collaborations include DreamCraft3D (Tsinghua/DeepSeek/
independent researcher), MoE and loss-free work (university collaborations), NSA
(DeepSeek/PKU/University of Washington), Prover (DeepSeek and universities), and
GRM/SPCT (DeepSeek/Tsinghua). Their author blocks were checked where stated.

Fire-Flyer v2 explicitly lists **DeepSeek-AI, Beijing**. It is not relabelled as
solely High-Flyer work just because HFAiLab/HAI-Platform appears in its historical
infrastructure lineage. Earlier HAI tools are separated below the main catalogue.
Original Hyper-Connections and Muon origins/scaling work remain external.

## Unresolved source and metadata gaps

- V4.1 report existence is confirmed by its official launch and HF file page.
  The file page identifies upload `53e70b1`, size 1.81 MB, and advertised SHA256
  `ba68e2e40408125ae6d2f63a9a241b61c73910691c74ec1a2a7023c851eac08d`.
  PDF resolve/download attempts returned 403 or a browser fetch error. This is
  **published but inaccessible**, not unpublished. Canonical PDF title and methods
  remain unread; no implementation is inferred from the launch's architecture name.
- V3.2-Exp's repository contains `DeepSeek_V3_2.pdf`; direct full-PDF fetching
  failed in this audit. The full report and corrected indexer positional code
  remain prerequisites for implementation. NSA is a distinct attention method.
- The retrieved V4 arXiv page for `2606.19348` reports v1 submission on 2026-04-26,
  inconsistent with the identifier's month. Its official release is dated Apr 24.
  The catalogue preserves observed metadata and flags the inconsistency; no date
  is silently invented or normalized to fit the identifier.
- Full HF enumeration, all repository-linked ancillary publications, non-arXiv
  collaborations and every employee's affiliation-at-publication history remain
  uncertain. Further discovery may add rows. API failures are not evidence of
  absence, and no inaccessible source is replaced with fabricated architecture.
