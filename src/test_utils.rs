use std::sync::Arc;

pub(crate) fn arc<I>(i: I) -> Arc<I> {
    Arc::new(i)
}
