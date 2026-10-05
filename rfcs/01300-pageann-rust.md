# PageANN in DiskANN3 (Rust)

| | |
|---|---|
| **Status** | Draft — implementation in progress on `pr_curs_add_pageann_to_diskann` |
| **Created** | 2026-10-05 |

## Summary

Port the **PageANN** algorithm (SSD page-packed graph + nav graph) into this DiskANN3 workspace as a new crate `diskann-pageann`, written in Rust. This is **not** the existing `paged_search` API (incremental result pages).

Source of truth for behavior: the C++ tree at `PageANN/` (`generate_page_graph`, `build_pageann_nav_graph`, `search_disk_index` with page layout).

## Naming

| Term | Meaning |
| --- | --- |
| `PagedSearch` / RFC 01078 | Return KNN hits in successive *result* pages without restarting graph walk |
| **PageANN** (this RFC) | Pack several vectors into a 4 KiB *SSD page*, build a **page-level** graph, search with page I/O |

The crate is named `diskann-pageann` so the two ideas stay distinct.

## Pipeline (C++ today)

1. Build a **vector-level Vamana disk index** (`build_vamana_disk_index` / `diskann-disk`).
2. **`generate_page_graph`**: cluster Vamana neighbors into mega-nodes (pages), rewrite IDs, write page-aligned layout + PQ.
3. **`build_pageann_nav_graph`**: small in-memory graph over sampled pages for entry points.
4. **Search**: PQ in memory → nav L hops → beam search over page graph (`W` beam, page cache).

Rust should keep the same stages. Step 1 already exists in `diskann-disk`. This crate owns 2–4.

## Non-goals (this crate, first milestones)

- Bind PageANN into Chroma (later, after search recall matches C++ on SIFT crop).
- Port LAANN / MegaANN batch-interleave (`build_laann_graph`).
- Change DiskANN3 `paged_search`.
- Billion-scale until layout + search match C++ on SIFT1M.

## Crate layout

```text
diskann-pageann/
  src/
    lib.rs          # crate docs, re-exports
    layout.rs       # 4 KiB page geometry, id maps (this milestone)
    pack.rs         # merge Vamana nodes into pages (next)
    nav.rs          # nav graph (later)
    search.rs       # page-graph beam search (later)
```

Depends on `diskann`, `diskann-disk`, `diskann-vector` — consume a built Vamana disk index rather than reimplementing Vamana.

## Milestone 0 (this change)

- RFC
- Workspace member `diskann-pageann`
- `layout`: SSD page size, vectors-per-page helper, `PageAnnLayout` + id-map types, unit tests

## Milestone 1 (this change)

- Read a tiny synthetic Vamana adjacency list
- Pack into pages with an explicit id map (`old_id → (page, slot)`)
- Sequential + greedy-neighbor packers (`pack.rs`); on-disk 4 KiB image still later

## Milestone 2

- Page-graph beam search vs brute force on a toy set (≥256 pts when wired to DiskANN build)
- Then SIFT 10K / 1M vs C++ PageANN Recall@10

## File / I/O notes from C++

C++ writes `*_PageANN` prefixes, `original_to_new_ids_map.bin`, PQ sidecars. Rust may use a directory + manifest (DiskANN3 style) instead of the C++ prefix soup, as long as the **geometry** (vectors per 4 KiB page, page degree ≈ R × vectors_per_page) matches.
