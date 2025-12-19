use bolt_autodiff::{Float, HasParams, Parameter};
use bolt_core::BaseBackend;

pub trait NamedParams<B, D>
where
    B: BaseBackend,
    D: Float,
{
    fn visit_named_params(&self, f: &mut dyn FnMut(&str, &Parameter<B, D>));
    fn visit_named_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B, D>));
}

pub struct FlatNamed<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    pub module: &'a T,
    _marker: std::marker::PhantomData<(B, D)>,
}

impl<'a, B, D, T> FlatNamed<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    pub fn new(module: &'a T) -> Self {
        Self {
            module,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, B, D, T> NamedParams<B, D> for FlatNamed<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    fn visit_named_params(&self, f: &mut dyn FnMut(&str, &Parameter<B, D>)) {
        let mut i = 0usize;
        self.module.visit_params(&mut |p| {
            let key = format!("p{}.{}", i, p.display_name());
            i += 1;
            f(&key, p);
        });
    }

    fn visit_named_params_mut(&mut self, _f: &mut dyn FnMut(&str, &mut Parameter<B, D>)) {
        unreachable!("FlatNamed is immutable; use FlatNamedMut")
    }
}

pub struct FlatNamedMut<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    pub module: &'a mut T,
    _marker: std::marker::PhantomData<(B, D)>,
}

impl<'a, B, D, T> FlatNamedMut<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    pub fn new(module: &'a mut T) -> Self {
        Self {
            module,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<'a, B, D, T> NamedParams<B, D> for FlatNamedMut<'a, B, D, T>
where
    B: BaseBackend,
    D: Float,
    T: HasParams<B, D> + ?Sized,
{
    fn visit_named_params(&self, f: &mut dyn FnMut(&str, &Parameter<B, D>)) {
        let mut i = 0usize;
        self.module.visit_params(&mut |p| {
            let key = format!("p{}.{}", i, p.display_name());
            i += 1;
            f(&key, p);
        });
    }

    fn visit_named_params_mut(&mut self, f: &mut dyn FnMut(&str, &mut Parameter<B, D>)) {
        let mut i = 0usize;
        self.module.visit_params_mut(&mut |p| {
            let key = format!("p{}.{}", i, p.display_name());
            i += 1;
            f(&key, p);
        });
    }
}

pub fn join_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_string()
    } else {
        format!("{}.{}", prefix, segment)
    }
}
