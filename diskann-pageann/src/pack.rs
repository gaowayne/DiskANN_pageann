//! Pack vector-level Vamana vertices into SSD pages.
//!
//! Two strategies:
//! - [`PackStrategy::Sequential`]: dense IDs in order (baseline, C++-unlike).
//! - [`PackStrategy::GreedyNeighbors`]: fill each page with a seed and its
//!   still-unassigned Vamana neighbors (topology clustering, closer to PageANN).

use crate::layout::{PageAnnLayout, PageId, PackedLocation, Slot};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PackStrategy {
    Sequential,
    GreedyNeighbors,
}

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum PackError {
    #[error("adjacency length {got} != num_vectors {expected}")]
    AdjacencyLen { got: usize, expected: u32 },
    #[error("neighbor id {id} out of range (n={n})")]
    NeighborOutOfRange { id: u32, n: u32 },
    #[error(transparent)]
    Layout(#[from] crate::layout::LayoutError),
}

/// Packed PageANN graph in memory (not yet a 4 KiB on-disk image).
#[derive(Clone, Debug)]
pub struct PackedPageGraph {
    pub layout: PageAnnLayout,
    /// `old_id → dense new_id` (new ids are packed: page * C + slot).
    pub old_to_new: Vec<u32>,
    /// `page → old ids` in slot order (last page may be short).
    pub pages: Vec<Vec<u32>>,
    /// `page → neighbor page ids` (deduped, excluding self).
    pub page_neighbors: Vec<Vec<u32>>,
}

impl PackedPageGraph {
    pub fn location_of_old_id(&self, old_id: u32) -> Option<PackedLocation> {
        let new = *self.old_to_new.get(old_id as usize)?;
        self.layout.location_of_dense_id(new)
    }
}

/// `adjacency[i]` is the Vamana neighbor list of old id `i`.
pub fn pack_vamana(
    adjacency: &[Vec<u32>],
    dim: usize,
    bytes_per_value: usize,
    graph_degree: usize,
    strategy: PackStrategy,
) -> Result<PackedPageGraph, PackError> {
    let n = adjacency.len() as u32;
    let layout = PageAnnLayout::from_params(dim, bytes_per_value, graph_degree, n)?;
    if adjacency.len() != n as usize {
        return Err(PackError::AdjacencyLen {
            got: adjacency.len(),
            expected: n,
        });
    }
    for (src, nbrs) in adjacency.iter().enumerate() {
        for &id in nbrs {
            if id >= n {
                return Err(PackError::NeighborOutOfRange { id, n });
            }
        }
        let _ = src;
    }

    let pages = match strategy {
        PackStrategy::Sequential => sequential_pages(n, layout.vectors_per_page),
        PackStrategy::GreedyNeighbors => greedy_neighbor_pages(adjacency, layout.vectors_per_page),
    };

    let mut old_to_new = vec![0u32; n as usize];
    for (page_idx, page) in pages.iter().enumerate() {
        for (slot, &old) in page.iter().enumerate() {
            let new_id = (page_idx as u32) * (layout.vectors_per_page as u32) + slot as u32;
            old_to_new[old as usize] = new_id;
        }
    }

    let page_neighbors = page_level_edges(&pages, adjacency);
    let mut layout = layout;
    layout.num_pages = pages.len() as u32;

    Ok(PackedPageGraph {
        layout,
        old_to_new,
        pages,
        page_neighbors,
    })
}

fn sequential_pages(n: u32, cap: usize) -> Vec<Vec<u32>> {
    let mut pages = Vec::new();
    let mut cur = Vec::with_capacity(cap);
    for id in 0..n {
        cur.push(id);
        if cur.len() == cap {
            pages.push(std::mem::take(&mut cur));
            cur = Vec::with_capacity(cap);
        }
    }
    if !cur.is_empty() {
        pages.push(cur);
    }
    pages
}

fn greedy_neighbor_pages(adjacency: &[Vec<u32>], cap: usize) -> Vec<Vec<u32>> {
    let n = adjacency.len();
    let mut assigned = vec![false; n];
    let mut pages = Vec::new();
    for seed in 0..n {
        if assigned[seed] {
            continue;
        }
        let mut page = Vec::with_capacity(cap);
        let mut stack = vec![seed as u32];
        while page.len() < cap {
            let Some(id) = stack.pop() else {
                break;
            };
            if assigned[id as usize] {
                continue;
            }
            assigned[id as usize] = true;
            page.push(id);
            for &nbr in adjacency[id as usize].iter().rev() {
                if !assigned[nbr as usize] {
                    stack.push(nbr);
                }
            }
        }
        // Fill leftover slots with the next unassigned ids so pages stay dense.
        if page.len() < cap {
            for id in 0..n {
                if page.len() == cap {
                    break;
                }
                if !assigned[id] {
                    assigned[id] = true;
                    page.push(id as u32);
                }
            }
        }
        pages.push(page);
    }
    pages
}

fn page_level_edges(pages: &[Vec<u32>], adjacency: &[Vec<u32>]) -> Vec<Vec<u32>> {
    let n = adjacency.len();
    let mut old_to_page = vec![0u32; n];
    for (p, page) in pages.iter().enumerate() {
        for &old in page {
            old_to_page[old as usize] = p as u32;
        }
    }
    pages
        .iter()
        .enumerate()
        .map(|(p, page)| {
            let mut nbrs = Vec::new();
            for &old in page {
                for &v in &adjacency[old as usize] {
                    let q = old_to_page[v as usize];
                    if q != p as u32 && !nbrs.contains(&q) {
                        nbrs.push(q);
                    }
                }
            }
            nbrs
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: u32, degree: usize) -> Vec<Vec<u32>> {
        (0..n)
            .map(|i| {
                (1..=degree)
                    .map(|d| (i + d as u32) % n)
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn sequential_covers_every_id_once() {
        let adj = ring(10, 2);
        let packed = pack_vamana(&adj, 8, 4, 2, PackStrategy::Sequential).unwrap();
        let mut seen = vec![false; 10];
        for page in &packed.pages {
            for &id in page {
                assert!(!seen[id as usize]);
                seen[id as usize] = true;
            }
        }
        assert!(seen.iter().all(|&b| b));
        assert_eq!(packed.location_of_old_id(0), Some(PackedLocation {
            page: PageId(0),
            slot: Slot(0),
        }));
    }

    #[test]
    fn greedy_prefers_neighbors_on_same_page() {
        // Two cliques of 3, weakly linked.
        let mut adj = vec![vec![]; 6];
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }
        for i in 3..6 {
            for j in 3..6 {
                if i != j {
                    adj[i].push(j as u32);
                }
            }
        }
        adj[2].push(3);
        adj[3].push(2);
        let packed = pack_vamana(&adj, 4, 4, 2, PackStrategy::GreedyNeighbors).unwrap();
        let p0 = packed.location_of_old_id(0).unwrap().page;
        let p1 = packed.location_of_old_id(1).unwrap().page;
        assert_eq!(p0, p1);
    }

    #[test]
    fn rejects_oob_neighbor() {
        let adj = vec![vec![99u32]];
        assert!(matches!(
            pack_vamana(&adj, 8, 4, 1, PackStrategy::Sequential),
            Err(PackError::NeighborOutOfRange { .. })
        ));
    }
}
