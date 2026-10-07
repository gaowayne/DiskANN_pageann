//! PageANN: pack DiskANN Vamana vertices into SSD pages and search a page graph.
//!
//! Distinct from [`diskann`] **paged search** (paginated KNN *results*).
//! See `rfcs/01300-pageann-rust.md`.

pub mod layout;
pub mod pack;
pub mod search;

pub use layout::{
    page_graph_degree, vectors_per_page, LayoutError, PackedLocation, PageAnnLayout, PageId,
    Slot, PAGE_HEADER_BYTES, SSD_PAGE_BYTES,
};
pub use pack::{pack_vamana, PackError, PackStrategy, PackedPageGraph};
pub use search::{
    brute_force_topk, knn_adjacency, l2_squared, page_beam_search, SearchError, SearchHit,
    SearchParams,
};
