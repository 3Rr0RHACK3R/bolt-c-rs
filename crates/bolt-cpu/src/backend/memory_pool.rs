use std::alloc::{self, Layout};
use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
struct Ptr(NonNull<u8>);

unsafe impl Send for Ptr {}
unsafe impl Sync for Ptr {}

/// A thread-safe memory pool that caches allocations to minimize overhead and fragmentation.
/// It buckets allocations by (size, alignment).
#[derive(Debug)]
pub struct MemoryPool {
    // Map from (size, align) -> Stack of pointers
    blocks: Mutex<HashMap<(usize, usize), Vec<Ptr>>>,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            blocks: Mutex::new(HashMap::new()),
        }
    }

    /// Acquires a memory block matching the requested layout.
    ///
    /// If a compatible block is in the pool, it is returned.
    /// Otherwise, a new block is allocated from the system.
    ///
    /// Returns a pointer to the block and the layout used for allocation (which matches the requested layout).
    pub fn acquire(&self, layout: Layout) -> NonNull<u8> {
        let key = (layout.size(), layout.align());

        // 1. Try to pop from cache
        {
            let mut blocks = self.blocks.lock().unwrap();
            if let Some(stack) = blocks.get_mut(&key) {
                if let Some(wrapper) = stack.pop() {
                    return wrapper.0;
                }
            }
        }

        // 2. Allocate from system if not found
        // Safety: Layout is checked by caller to be non-zero usually,
        // but alloc requires non-zero size.
        let size = layout.size();
        if size == 0 {
            return NonNull::dangling();
        }

        unsafe {
            let ptr = alloc::alloc(layout);
            NonNull::new(ptr).unwrap_or_else(|| alloc::handle_alloc_error(layout))
        }
    }

    /// Releases a memory block back into the pool.
    ///
    /// # Safety
    /// The `ptr` must have been allocated with the given `layout`.
    /// The caller must ensure that the memory is no longer used.
    pub unsafe fn release(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let key = (layout.size(), layout.align());
        let mut blocks = self.blocks.lock().unwrap();
        blocks.entry(key).or_default().push(Ptr(ptr));
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        let blocks = self.blocks.get_mut().unwrap();
        for ((size, align), stack) in blocks.iter_mut() {
            let layout = Layout::from_size_align(*size, *align).unwrap();
            for wrapper in stack.drain(..) {
                unsafe {
                    alloc::dealloc(wrapper.0.as_ptr(), layout);
                }
            }
        }
    }
}
