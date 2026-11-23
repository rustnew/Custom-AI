/// Les blocs alignés gardent les mêmes calculs de préremplissage avec et sans cache.
pub const PREFILL_BLOCK_SIZE: usize = 128;

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
        let mut cache = PrefixCache::new(256);
        let prefix: Vec<_> = (0..128).collect();
        cache.store(&prefix, "state");
        let mut next = prefix.clone();
        next.extend([7, 8]);
        assert_eq!(cache.lookup(&next), Some((128, &"state")));
        next[17] = 999;
        assert!(cache.lookup(&next).is_none());
        assert!(cache.lookup(&prefix[..127]).is_none());
        cache.clear();
        assert!(cache.lookup(&prefix).is_none());
    }

    #[test]
    fn bounds_memory_and_supports_a_disabled_cache() {
        let mut cache = PrefixCache::new(256);
        cache.store(&vec![1; 384], "too-large");
        assert!(cache.lookup(&vec![1; 512]).is_none());
        assert_eq!(cache.target_len(10000), 256);
        assert_eq!(cache.target_len(127), 0);
        let mut disabled = PrefixCache::new(0);
        disabled.store(&[1; 128], "disabled");
        assert!(disabled.lookup(&[1; 128]).is_none());
    }
}
