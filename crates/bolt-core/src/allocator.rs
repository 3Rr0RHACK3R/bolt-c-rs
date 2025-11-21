use crate::{
    dtype::{DType, NativeType},
    error::{Error, Result},
};

pub trait StorageAllocator<D: NativeType>: Clone + Send + Sync + 'static {
    type Storage: Clone + Send + Sync + 'static;

    fn allocate(&self, len: usize) -> Result<Self::Storage>;
    fn allocate_zeroed(&self, len: usize) -> Result<Self::Storage>;

    fn allocate_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let elem_size = dtype.size_in_bytes();
        if !len_bytes.is_multiple_of(elem_size) {
            return Err(Error::invalid_shape(format!(
                "byte length {} is not aligned to dtype size {}",
                len_bytes, elem_size
            )));
        }
        self.allocate(len_bytes / elem_size)
    }

    fn allocate_zeroed_bytes(&self, len_bytes: usize, dtype: DType) -> Result<Self::Storage> {
        let elem_size = dtype.size_in_bytes();
        if !len_bytes.is_multiple_of(elem_size) {
            return Err(Error::invalid_shape(format!(
                "byte length {} is not aligned to dtype size {}",
                len_bytes, elem_size
            )));
        }
        self.allocate_zeroed(len_bytes / elem_size)
    }

    fn release(&self, _storage: Self::Storage) {}
}
