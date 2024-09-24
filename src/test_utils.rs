use std::sync::{Arc, RwLock};

pub(crate) fn arc<I>(i: I) -> Arc<RwLock<I>> {
    Arc::new(RwLock::new(i))
}
