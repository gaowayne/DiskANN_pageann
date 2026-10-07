//! In-memory PageANN beam search over a packed page graph.
//!
//! Mirrors the C++ `page_search` control flow without PQ or SSD I/O yet:
//! visit whole pages, score every vector on the page at full precision, then
//! expand unvisited neighbor pages with beam width `W`.

use crate::pack::PackedPageGraph;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct SearchParams {
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Candidate list size (DiskANN / PageANN `L`).
    pub l_search: usize,
    /// Max pages expanded per iteration (`beam_width` / `W`).
    pub beam_width: usize,
}

impl Default for SearchParams {
    fn default() -> Self {
        Self {
            k: 10,
            l_search: 50,
            beam_width: 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchHit {
    pub id: u32,
    pub distance: f32,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum SearchError {
    #[error("query dim {got} != index dim {expected}")]
    DimMismatch { got: usize, expected: usize },
    #[error("vector buffer length {got} != num_vectors*{dim} ({expected})")]
    VectorLen { got: usize, expected: usize },
    #[error("entry page {page} out of range (num_pages={num_pages})")]
    EntryPage { page: u32, num_pages: u32 },
    #[error("k and l_search must be positive")]
    InvalidParams,
}

#[derive(Clone, Copy)]
struct Cand {
    dist: f32,
    page: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.page == other.page && self.dist == other.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap via reverse order on distance.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.page.cmp(&other.page))
    }
}

#[derive(Clone, Copy)]
struct Hit {
    dist: f32,
    id: u32,
}

impl PartialEq for Hit {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist == other.dist
    }
}
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Hit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

/// Squared L2 distance between `query` and `vectors[id * dim ..]`.
pub fn l2_squared(query: &[f32], vectors: &[f32], id: u32, dim: usize) -> f32 {
    let start = id as usize * dim;
    let row = &vectors[start..start + dim];
    let mut sum = 0.0f32;
    for (a, b) in query.iter().zip(row.iter()) {
        let d = a - b;
        sum += d * d;
    }
    sum
}

/// Brute-force top-`k` by squared L2 (old ids = row index).
pub fn brute_force_topk(query: &[f32], vectors: &[f32], dim: usize, k: usize) -> Vec<SearchHit> {
    let n = vectors.len() / dim;
    let mut hits: Vec<Hit> = (0..n as u32)
        .map(|id| Hit {
            dist: l2_squared(query, vectors, id, dim),
            id,
        })
        .collect();
    hits.sort();
    hits.truncate(k.min(n));
    hits.into_iter()
        .map(|h| SearchHit {
            id: h.id,
            distance: h.dist,
        })
        .collect()
}

/// Page-graph beam search. `vectors` is row-major `num_vectors × dim` in **old id** order.
pub fn page_beam_search(
    graph: &PackedPageGraph,
    vectors: &[f32],
    query: &[f32],
    entry_page: u32,
    params: &SearchParams,
) -> Result<Vec<SearchHit>, SearchError> {
    let dim = graph.layout.dim;
    if query.len() != dim {
        return Err(SearchError::DimMismatch {
            got: query.len(),
            expected: dim,
        });
    }
    let expected = graph.layout.num_vectors as usize * dim;
    if vectors.len() != expected {
        return Err(SearchError::VectorLen {
            got: vectors.len(),
            expected,
        });
    }
    if entry_page >= graph.layout.num_pages {
        return Err(SearchError::EntryPage {
            page: entry_page,
            num_pages: graph.layout.num_pages,
        });
    }
    if params.k == 0 || params.l_search == 0 || params.beam_width == 0 {
        return Err(SearchError::InvalidParams);
    }

    let mut visited: HashSet<u32> = HashSet::new();
    let mut frontier: BinaryHeap<Cand> = BinaryHeap::new();
    frontier.push(Cand {
        dist: 0.0,
        page: entry_page,
    });

    // Best vectors seen so far (sorted ascending by distance).
    let mut best: Vec<Hit> = Vec::with_capacity(params.l_search);
    // Cap page visits similarly to DiskANN `L` / beam exploration.
    let max_pages = params.l_search.max(params.beam_width);

    while visited.len() < max_pages {
        // Take up to beam_width closest unvisited pages this round.
        let mut batch: Vec<u32> = Vec::with_capacity(params.beam_width);
        while batch.len() < params.beam_width {
            let Some(Cand { page, .. }) = frontier.pop() else {
                break;
            };
            if visited.contains(&page) {
                continue;
            }
            batch.push(page);
        }
        if batch.is_empty() {
            break;
        }

        for page in batch {
            if !visited.insert(page) {
                continue;
            }

            let page_vecs = &graph.pages[page as usize];
            let mut page_best = f32::INFINITY;
            for &old_id in page_vecs {
                let dist = l2_squared(query, vectors, old_id, dim);
                page_best = page_best.min(dist);
                insert_hit(&mut best, Hit { dist, id: old_id }, params.l_search);
            }

            for &nbr in &graph.page_neighbors[page as usize] {
                if visited.contains(&nbr) {
                    continue;
                }
                frontier.push(Cand {
                    dist: page_best,
                    page: nbr,
                });
            }
        }

        // Keep frontier bounded.
        if frontier.len() > params.beam_width * params.l_search {
            let mut keep: Vec<Cand> = frontier.drain().collect();
            keep.sort_by(|a, b| {
                a.dist
                    .partial_cmp(&b.dist)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.page.cmp(&b.page))
            });
            keep.truncate(params.beam_width * params.l_search);
            frontier.extend(keep);
        }
    }

    best.truncate(params.k.min(best.len()));
    Ok(best
        .into_iter()
        .map(|h| SearchHit {
            id: h.id,
            distance: h.dist,
        })
        .collect())
}

fn insert_hit(best: &mut Vec<Hit>, hit: Hit, cap: usize) {
    if best.iter().any(|h| h.id == hit.id) {
        return;
    }
    match best.binary_search(&hit) {
        Ok(i) | Err(i) => best.insert(i, hit),
    }
    if best.len() > cap {
        best.truncate(cap);
    }
}

/// Build a simple exact-kNN adjacency (each node → `degree` nearest others by L2).
pub fn knn_adjacency(vectors: &[f32], dim: usize, degree: usize) -> Vec<Vec<u32>> {
    let n = vectors.len() / dim;
    let deg = degree.min(n.saturating_sub(1));
    (0..n)
        .map(|i| {
            let mut nbrs: Vec<(f32, u32)> = (0..n as u32)
                .filter(|&j| j != i as u32)
                .map(|j| {
                    (
                        l2_squared(&vectors[i * dim..(i + 1) * dim], vectors, j, dim),
                        j,
                    )
                })
                .collect();
            nbrs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            nbrs.truncate(deg);
            nbrs.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::{pack_vamana, PackStrategy};

    fn recall_at_k(got: &[SearchHit], truth: &[SearchHit]) -> f32 {
        let truth_ids: HashSet<u32> = truth.iter().map(|h| h.id).collect();
        let hits = got.iter().filter(|h| truth_ids.contains(&h.id)).count();
        hits as f32 / truth.len().max(1) as f32
    }

    #[test]
    fn page_search_high_recall_on_toy_knn_graph() {
        let dim = 8;
        let n = 64usize;
        let degree = 8;
        // Deterministic pseudo-random vectors.
        let mut vectors = vec![0.0f32; n * dim];
        for i in 0..n {
            for d in 0..dim {
                vectors[i * dim + d] = ((i * 17 + d * 31) % 100) as f32 / 50.0;
            }
        }
        let adj = knn_adjacency(&vectors, dim, degree);
        let graph = pack_vamana(&adj, dim, 4, degree, PackStrategy::GreedyNeighbors).unwrap();

        let params = SearchParams {
            k: 10,
            l_search: 40,
            beam_width: 4,
        };
        let mut sum = 0.0f32;
        let queries = 16usize;
        for q in 0..queries {
            let query: Vec<f32> = (0..dim)
                .map(|d| ((q * 13 + d * 7) % 100) as f32 / 50.0)
                .collect();
            let truth = brute_force_topk(&query, &vectors, dim, params.k);
            let got = page_beam_search(&graph, &vectors, &query, 0, &params).unwrap();
            sum += recall_at_k(&got, &truth);
        }
        let mean = sum / queries as f32;
        assert!(mean >= 0.85, "mean Recall@10 = {mean}, expected >= 0.85");
    }

    #[test]
    fn rejects_dim_mismatch() {
        let adj = vec![vec![1u32], vec![0u32]];
        let graph = pack_vamana(&adj, 4, 4, 1, PackStrategy::Sequential).unwrap();
        let vectors = vec![0.0f32; 8];
        let err = page_beam_search(
            &graph,
            &vectors,
            &[0.0, 0.0],
            0,
            &SearchParams::default(),
        )
        .unwrap_err();
        assert!(matches!(err, SearchError::DimMismatch { .. }));
    }
}
