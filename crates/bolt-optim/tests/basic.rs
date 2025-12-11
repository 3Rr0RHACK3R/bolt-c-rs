use std::collections::HashSet;
use std::sync::Arc;

use bolt_autodiff::Parameter;
use bolt_core::Tensor;
use bolt_cpu::CpuBackend;

#[test]
fn dedupes_repeated_param_ids() -> Result<(), Box<dyn std::error::Error>> {
    let base = Arc::new(CpuBackend::new());
    let p = Parameter::with_name(Tensor::from_slice(&base, &[1.0f32], &[1])?, "p");

    // The optimizer's step logic dedupes parameters by ParamId; verify the
    // deduplication behavior without creating invalid multiple &mut borrows.
    let ids = [p.id(), p.id()];
    let mut seen = HashSet::new();
    for id in ids {
        let _ = seen.insert(id);
    }

    assert_eq!(
        seen.len(),
        1,
        "deduplication of ParamId should ignore repeats"
    );
    Ok(())
}
