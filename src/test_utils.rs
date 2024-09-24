use std::sync::{Arc, RwLock};

#[allow(dead_code)]
pub(crate) fn arc<I>(i: I) -> Arc<RwLock<I>> {
    Arc::new(RwLock::new(i))
}
