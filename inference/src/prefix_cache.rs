/// Les blocs alignés gardent les mêmes calculs de préremplissage avec et sans cache.
pub const PREFILL_BLOCK_SIZE: usize = 512;

/// Les grands contextes gardent des matrices d'attention temporaires plus petites.
/// Les deux tailles divisent l'alignement des snapshots de préfixe.
pub fn prefill_block_size(prompt_tokens: usize) -> usize {
    if prompt_tokens > 16384 {
        128
    } else {
        PREFILL_BLOCK_SIZE
    }
}

pub struct PrefixCache<T> {
    capacity: usize,
    entry: Option<(Vec<u32>, T)>,
}

impl<T> PrefixCache<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entry: None,
        }
    }

    pub fn target_len(&self, prompt_len: usize) -> usize {
        prompt_len.min(self.capacity) / PREFILL_BLOCK_SIZE * PREFILL_BLOCK_SIZE
    }

    pub fn lookup(&self, prompt: &[u32]) -> Option<(usize, &T)> {
        let (tokens, state) = self.entry.as_ref()?;
        prompt.starts_with(tokens).then_some((tokens.len(), state))
    }

    pub fn store(&mut self, tokens: &[u32], state: T) {
        if !tokens.is_empty()
            && tokens.len() <= self.capacity
            && tokens.len().is_multiple_of(PREFILL_BLOCK_SIZE)
        {
            self.entry = Some((tokens.to_vec(), state));
        }
    }

    pub fn clear(&mut self) {
        self.entry = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuses_only_an_exact_complete_prefix_without_crossing_request_boundaries() {
        let mut cache = PrefixCache::new(2 * PREFILL_BLOCK_SIZE);
        let prefix: Vec<_> = (0..PREFILL_BLOCK_SIZE as u32).collect();
        cache.store(&prefix, "state");
        let mut next = prefix.clone();
        next.extend([7, 8]);
        assert_eq!(cache.lookup(&next), Some((PREFILL_BLOCK_SIZE, &"state")));
        next[17] = 999;
        assert!(cache.lookup(&next).is_none());
        assert!(cache.lookup(&prefix[..PREFILL_BLOCK_SIZE - 1]).is_none());
        cache.clear();
        assert!(cache.lookup(&prefix).is_none());
    }

    #[test]
    fn bounds_memory_and_supports_a_disabled_cache() {
        let mut cache = PrefixCache::new(2 * PREFILL_BLOCK_SIZE);
        cache.store(&vec![1; 3 * PREFILL_BLOCK_SIZE], "too-large");
        assert!(cache.lookup(&vec![1; 4 * PREFILL_BLOCK_SIZE]).is_none());
        assert_eq!(cache.target_len(10000), 2 * PREFILL_BLOCK_SIZE);
        assert_eq!(cache.target_len(PREFILL_BLOCK_SIZE - 1), 0);
        let mut disabled = PrefixCache::new(0);
        disabled.store(&[1; PREFILL_BLOCK_SIZE], "disabled");
        assert!(disabled.lookup(&[1; PREFILL_BLOCK_SIZE]).is_none());
    }

    #[test]
    fn every_prefill_size_can_reach_prefix_snapshot_boundaries() {
        assert_eq!(prefill_block_size(16384), 512);
        assert_eq!(prefill_block_size(16385), 128);
        for prompt in [1024, 3616, 15360, 16384, 16385, 28535] {
            let block = prefill_block_size(prompt);
            let cache = PrefixCache::<()>::new(7680);
            assert!(cache.target_len(prompt).is_multiple_of(block));
        }
    }
}
