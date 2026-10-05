//! SSD page geometry and ID maps for PageANN.
//!
//! A PageANN **page** is a disk-aligned block (default 4 KiB) that stores several
//! full-precision vectors plus a page-level adjacency list. Search I/O is in
//! whole pages, unlike vector-level DiskANN which fetches one vertex record.
//!
//! Vector-level Vamana IDs are remapped: `old_id → (page_id, slot)`.

use serde::{Deserialize, Serialize};

/// Typical NVMe/SSD logical block used by the C++ PageANN layout.
pub const SSD_PAGE_BYTES: usize = 4096;

/// Bytes reserved in each page for header / padding (kept conservative).
pub const PAGE_HEADER_BYTES: usize = 32;

#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum LayoutError {
    #[error("dimension must be positive")]
    InvalidDimension,
    #[error("page cannot hold any vector of dim {dim} ({bytes_per_value} bytes/value)")]
    PageTooSmall { dim: usize, bytes_per_value: usize },
    #[error("graph degree must be positive")]
    InvalidDegree,
}

/// How many float32 (or same-width) vectors fit in one page if each also
/// stores `graph_degree` neighbor IDs (`u32`).
///
/// Matches the PageANN rule of thumb: page degree ≈ R × vectors_per_page,
/// so `graph_degree` here is the **per-vector** Vamana R (not the page degree).
pub fn vectors_per_page(
    dim: usize,
    bytes_per_value: usize,
    graph_degree: usize,
) -> Result<usize, LayoutError> {
    if dim == 0 {
        return Err(LayoutError::InvalidDimension);
    }
    if bytes_per_value == 0 {
        return Err(LayoutError::PageTooSmall { dim, bytes_per_value });
    }
    if graph_degree == 0 {
        return Err(LayoutError::InvalidDegree);
    }
    let payload = SSD_PAGE_BYTES.saturating_sub(PAGE_HEADER_BYTES);
    let per_vector = dim
        .saturating_mul(bytes_per_value)
        .saturating_add(graph_degree.saturating_mul(4));
    if per_vector == 0 || per_vector > payload {
        return Err(LayoutError::PageTooSmall { dim, bytes_per_value });
    }
    Ok(payload / per_vector)
}

/// Page-level graph degree if each of `n_per_page` vectors keeps `graph_degree` neighbors.
pub fn page_graph_degree(n_per_page: usize, graph_degree: usize) -> usize {
    n_per_page.saturating_mul(graph_degree)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct PageId(pub u32);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Slot(pub u16);

/// Location of a vector-level ID after packing.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PackedLocation {
    pub page: PageId,
    pub slot: Slot,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PageAnnLayout {
    pub dim: usize,
    pub bytes_per_value: usize,
    pub graph_degree: usize,
    pub vectors_per_page: usize,
    pub page_graph_degree: usize,
    pub num_vectors: u32,
    pub num_pages: u32,
}

impl PageAnnLayout {
    pub fn from_params(
        dim: usize,
        bytes_per_value: usize,
        graph_degree: usize,
        num_vectors: u32,
    ) -> Result<Self, LayoutError> {
        let npp = vectors_per_page(dim, bytes_per_value, graph_degree)?;
        let num_pages = num_vectors.div_ceil(npp as u32);
        Ok(Self {
            dim,
            bytes_per_value,
            graph_degree,
            vectors_per_page: npp,
            page_graph_degree: page_graph_degree(npp, graph_degree),
            num_vectors,
            num_pages,
        })
    }

    pub fn location_of_dense_id(&self, new_id: u32) -> Option<PackedLocation> {
        if new_id >= self.num_vectors {
            return None;
        }
        let npp = self.vectors_per_page as u32;
        Some(PackedLocation {
            page: PageId(new_id / npp),
            slot: Slot((new_id % npp) as u16),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sift_like_page_holds_several_vectors() {
        // 128-d f32, R=32 → more than one vector per 4 KiB page.
        let n = vectors_per_page(128, 4, 32).unwrap();
        assert!(n >= 2, "expected packed page, got {n}");
        assert!(n <= 16);
    }

    #[test]
    fn layout_maps_dense_ids_into_pages() {
        let layout = PageAnnLayout::from_params(8, 4, 4, 10).unwrap();
        assert_eq!(layout.num_pages, 10u32.div_ceil(layout.vectors_per_page as u32));
        assert_eq!(
            layout.location_of_dense_id(0),
            Some(PackedLocation {
                page: PageId(0),
                slot: Slot(0),
            })
        );
        assert!(layout.location_of_dense_id(10).is_none());
    }

    #[test]
    fn rejects_zero_dim() {
        assert!(matches!(
            vectors_per_page(0, 4, 32),
            Err(LayoutError::InvalidDimension)
        ));
    }
}
