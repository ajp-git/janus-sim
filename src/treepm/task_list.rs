//! P2P task list with i_leaf-grouped indexing for block-per-group GPU dispatch.
//!
//! Reference: PhotoNs-GPU §3.2, Wang & Meng 2021.
//!
//! After tree construction, derive a list of leaf-pair tasks `<i_leaf, j_leaf>`
//! representing all pairs that must interact in P2P (short-range). The list
//! is then sorted by `i_leaf` so all tasks for a given i_leaf are contiguous,
//! and an index `GroupIdx` gives `(start, length)` per i_leaf.
//!
//! This enables a CUDA pattern where:
//! - 1 block = 1 i_leaf group
//! - 1 thread = 1 particle in i_leaf
//! - j_leaves prefetched in shared memory by chunks
//!
//! Result: maximizes register reuse for `i` data and shared-memory pre-fetch
//! for `j` neighbors.

/// One P2P interaction task: leaf pair (i, j) within r_cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P2PTask {
    pub i_leaf: u32,
    pub j_leaf: u32,
}

/// Indexing structure pointing into a sorted task list.
///
/// For each unique `i_leaf` value, gives the (start, length) of the contiguous
/// range of tasks in the sorted list.
#[derive(Debug, Clone)]
pub struct GroupIdx {
    pub i_leaves: Vec<u32>, // unique i_leaf values, sorted
    pub starts: Vec<u32>,   // start offset in tasks array per i_leaf
    pub lengths: Vec<u32>,  // number of j_leaves for this i_leaf
}

impl GroupIdx {
    pub fn n_groups(&self) -> usize {
        self.i_leaves.len()
    }
}

/// Build GroupIdx from a task list already sorted by i_leaf.
///
/// O(n) single-pass scan: detects boundaries where i_leaf changes.
pub fn build_group_idx(tasks: &[P2PTask]) -> GroupIdx {
    let mut i_leaves = Vec::new();
    let mut starts = Vec::new();
    let mut lengths = Vec::new();

    if tasks.is_empty() {
        return GroupIdx {
            i_leaves,
            starts,
            lengths,
        };
    }

    let mut current_leaf = tasks[0].i_leaf;
    let mut current_start: u32 = 0;
    for (k, t) in tasks.iter().enumerate() {
        if t.i_leaf != current_leaf {
            // Close previous group
            i_leaves.push(current_leaf);
            starts.push(current_start);
            lengths.push(k as u32 - current_start);
            // Open new group
            current_leaf = t.i_leaf;
            current_start = k as u32;
        }
    }
    // Close last group
    i_leaves.push(current_leaf);
    starts.push(current_start);
    lengths.push(tasks.len() as u32 - current_start);

    GroupIdx {
        i_leaves,
        starts,
        lengths,
    }
}

/// Sort task list by i_leaf in place.
///
/// Uses unstable sort (rayon::par_sort_unstable on host for large n).
pub fn sort_tasks_by_i_leaf(tasks: &mut [P2PTask]) {
    tasks.sort_unstable_by_key(|t| t.i_leaf);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_tasks() -> Vec<P2PTask> {
        // 5 leaves, varying neighbor counts
        vec![
            P2PTask { i_leaf: 2, j_leaf: 0 },
            P2PTask { i_leaf: 2, j_leaf: 1 },
            P2PTask { i_leaf: 2, j_leaf: 3 },
            P2PTask { i_leaf: 0, j_leaf: 1 },
            P2PTask { i_leaf: 0, j_leaf: 2 },
            P2PTask { i_leaf: 1, j_leaf: 0 },
            P2PTask { i_leaf: 1, j_leaf: 2 },
            P2PTask { i_leaf: 3, j_leaf: 0 },
            P2PTask { i_leaf: 3, j_leaf: 4 },
            P2PTask { i_leaf: 4, j_leaf: 1 },
        ]
    }

    #[test]
    fn test_sort_tasks_by_i_leaf() {
        let mut tasks = mock_tasks();
        sort_tasks_by_i_leaf(&mut tasks);
        for w in tasks.windows(2) {
            assert!(
                w[0].i_leaf <= w[1].i_leaf,
                "Tasks not sorted: {:?} > {:?}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_build_group_idx_consistency() {
        let mut tasks = mock_tasks();
        sort_tasks_by_i_leaf(&mut tasks);
        let idx = build_group_idx(&tasks);

        // Sanity: 5 unique i_leaves
        assert_eq!(idx.n_groups(), 5);
        assert_eq!(idx.i_leaves.len(), idx.starts.len());
        assert_eq!(idx.i_leaves.len(), idx.lengths.len());

        // For each group, verify all tasks have the right i_leaf
        for k in 0..idx.n_groups() {
            let start = idx.starts[k] as usize;
            let len = idx.lengths[k] as usize;
            let i_expected = idx.i_leaves[k];
            for offset in 0..len {
                assert_eq!(
                    tasks[start + offset].i_leaf, i_expected,
                    "Group {} task {} has wrong i_leaf",
                    k, offset
                );
            }
        }

        // Sum of lengths == total tasks
        let total: u32 = idx.lengths.iter().sum();
        assert_eq!(total as usize, tasks.len());
    }

    #[test]
    fn test_build_group_idx_empty() {
        let tasks: Vec<P2PTask> = vec![];
        let idx = build_group_idx(&tasks);
        assert_eq!(idx.n_groups(), 0);
    }

    #[test]
    fn test_build_group_idx_single_leaf() {
        let tasks = vec![
            P2PTask { i_leaf: 7, j_leaf: 0 },
            P2PTask { i_leaf: 7, j_leaf: 1 },
            P2PTask { i_leaf: 7, j_leaf: 2 },
        ];
        let idx = build_group_idx(&tasks);
        assert_eq!(idx.n_groups(), 1);
        assert_eq!(idx.i_leaves[0], 7);
        assert_eq!(idx.starts[0], 0);
        assert_eq!(idx.lengths[0], 3);
    }

    #[test]
    fn test_build_group_idx_sorted_consecutive_starts() {
        let mut tasks = mock_tasks();
        sort_tasks_by_i_leaf(&mut tasks);
        let idx = build_group_idx(&tasks);
        // starts must be monotonically increasing
        for w in idx.starts.windows(2) {
            assert!(w[0] < w[1], "starts not increasing: {} > {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_p2ptask_size() {
        // Verify P2PTask fits in 8 bytes (2 × u32) for cache-line packing
        assert_eq!(std::mem::size_of::<P2PTask>(), 8);
    }
}
