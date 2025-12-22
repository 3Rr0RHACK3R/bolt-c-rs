use bytemuck::cast;

use bolt_core::{
    dtype::{DType, NativeType},
    error::{Error, Result},
};

pub(crate) fn ensure_float_dtype(dtype: DType) -> Result<()> {
    if dtype.is_float() {
        Ok(())
    } else {
        Err(Error::invalid_shape(
            "operation requires floating point dtype",
        ))
    }
}

pub(crate) fn compute_arange_len<D>(start: D, end: D, step: D) -> Result<usize>
where
    D: NativeType,
{
    match D::DTYPE {
        DType::F32 => float_arange_len(
            cast::<D, f32>(start) as f64,
            cast::<D, f32>(end) as f64,
            cast::<D, f32>(step) as f64,
        )
        .and_then(float_len_to_usize),
        DType::F64 => float_arange_len(
            cast::<D, f64>(start),
            cast::<D, f64>(end),
            cast::<D, f64>(step),
        )
        .and_then(float_len_to_usize),
        DType::I32 => int_arange_len(
            cast::<D, i32>(start),
            cast::<D, i32>(end),
            cast::<D, i32>(step),
        ),
        DType::I64 => int_arange_len(
            cast::<D, i64>(start) as i32,
            cast::<D, i64>(end) as i32,
            cast::<D, i64>(step) as i32,
        ),
        DType::U8 => int_arange_len(
            cast::<D, u8>(start) as i32,
            cast::<D, u8>(end) as i32,
            cast::<D, u8>(step) as i32,
        ),
    }
}

fn float_len_to_usize(len: u128) -> Result<usize> {
    usize::try_from(len).map_err(|_| Error::TensorTooLarge {
        limit: isize::MAX as usize,
        requested: usize::MAX,
    })
}

pub(crate) fn build_arange_values<D>(len: usize, start: D, step: D) -> Result<Vec<D>>
where
    D: NativeType,
{
    match D::DTYPE {
        DType::F32 => {
            let start = cast::<D, f32>(start);
            let step = cast::<D, f32>(step);
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                values.push(cast(start + step * idx as f32));
            }
            Ok(values)
        }
        DType::F64 => {
            let start = cast::<D, f64>(start);
            let step = cast::<D, f64>(step);
            let mut values = Vec::with_capacity(len);
            for idx in 0..len {
                values.push(cast(start + step * idx as f64));
            }
            Ok(values)
        }
        DType::I32 => {
            let start_i64 = cast::<D, i32>(start) as i64;
            let step_i64 = cast::<D, i32>(step) as i64;
            let mut current = start_i64;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let value = i32::try_from(current)
                    .map_err(|_| Error::invalid_shape("arange value overflows i32 range"))?;
                values.push(cast(value));
                current = current.checked_add(step_i64).ok_or_else(|| {
                    Error::invalid_shape("arange value overflow during iteration")
                })?;
            }
            Ok(values)
        }
        DType::I64 => {
            let start_i128 = cast::<D, i64>(start) as i128;
            let step_i128 = cast::<D, i64>(step) as i128;
            let mut current = start_i128;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let value = i64::try_from(current)
                    .map_err(|_| Error::invalid_shape("arange value overflows i64 range"))?;
                values.push(cast(value));
                current = current.checked_add(step_i128).ok_or_else(|| {
                    Error::invalid_shape("arange value overflow during iteration")
                })?;
            }
            Ok(values)
        }
        DType::U8 => {
            let start_u16 = cast::<D, u8>(start) as u16;
            let step_u16 = cast::<D, u8>(step) as u16;
            let mut current = start_u16;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                let value = u8::try_from(current)
                    .map_err(|_| Error::invalid_shape("arange value overflows u8 range"))?;
                values.push(cast(value));
                current = current.checked_add(step_u16).ok_or_else(|| {
                    Error::invalid_shape("arange value overflow during iteration")
                })?;
            }
            Ok(values)
        }
    }
}

pub(crate) fn build_linspace_values<D>(start: D, end: D, steps: usize) -> Result<Vec<D>>
where
    D: NativeType,
{
    ensure_float_dtype(D::DTYPE)?;
    if steps < 2 {
        return Err(Error::invalid_shape("linspace steps must be >= 2"));
    }
    match D::DTYPE {
        DType::F32 => {
            let values = build_linspace_f32(cast::<D, f32>(start), cast::<D, f32>(end), steps);
            Ok(values.into_iter().map(cast).collect())
        }
        DType::F64 => {
            let values = build_linspace_f64(cast::<D, f64>(start), cast::<D, f64>(end), steps);
            Ok(values.into_iter().map(cast).collect())
        }
        DType::I32 | DType::I64 | DType::U8 => unreachable!(),
    }
}

pub(crate) fn build_logspace_values<D>(start: D, end: D, steps: usize, base: D) -> Result<Vec<D>>
where
    D: NativeType,
{
    ensure_float_dtype(D::DTYPE)?;
    if steps < 2 {
        return Err(Error::invalid_shape("logspace steps must be >= 2"));
    }
    match D::DTYPE {
        DType::F32 => {
            let values = build_logspace_f32(
                cast::<D, f32>(start),
                cast::<D, f32>(end),
                steps,
                cast::<D, f32>(base),
            )?;
            Ok(values.into_iter().map(cast).collect())
        }
        DType::F64 => {
            let values = build_logspace_f64(
                cast::<D, f64>(start),
                cast::<D, f64>(end),
                steps,
                cast::<D, f64>(base),
            )?;
            Ok(values.into_iter().map(cast).collect())
        }
        DType::I32 | DType::I64 | DType::U8 => unreachable!(),
    }
}

fn float_arange_len(start: f64, end: f64, step: f64) -> Result<u128> {
    if step == 0.0 || step.is_nan() {
        return Err(Error::invalid_shape(
            "arange step must be non-zero and not NaN",
        ));
    }
    if start.is_nan() || end.is_nan() {
        return Err(Error::invalid_shape("arange start/end must not be NaN"));
    }
    let span = end - start;
    let len = (span / step).ceil();
    if !len.is_finite() || len <= 0.0 {
        return Err(Error::invalid_shape(
            "arange would produce zero or infinite elements",
        ));
    }
    if len > isize::MAX as f64 {
        return Err(Error::TensorTooLarge {
            limit: isize::MAX as usize,
            requested: usize::MAX,
        });
    }
    let len_int = len as i128;
    if len_int <= 0 {
        return Err(Error::invalid_shape("arange would produce zero elements"));
    }
    u128::try_from(len_int).map_err(|_| Error::TensorTooLarge {
        limit: isize::MAX as usize,
        requested: usize::MAX,
    })
}

fn int_arange_len(start: i32, end: i32, step: i32) -> Result<usize> {
    if step == 0 {
        return Err(Error::invalid_shape("arange step must be non-zero"));
    }
    let delta = (end as i64) - (start as i64);
    let step_i64 = step as i64;
    if delta == 0 {
        return Err(Error::invalid_shape("arange would produce zero elements"));
    }
    if (delta > 0 && step_i64 <= 0) || (delta < 0 && step_i64 >= 0) {
        return Err(Error::invalid_shape(
            "arange step does not progress toward end",
        ));
    }
    let abs_delta = delta.abs() as i128;
    let abs_step = step_i64.unsigned_abs() as i128;
    let len = (abs_delta + (abs_step - 1)) / abs_step;
    if len == 0 {
        return Err(Error::invalid_shape("arange would produce zero elements"));
    }
    usize::try_from(len).map_err(|_| Error::TensorTooLarge {
        limit: isize::MAX as usize,
        requested: usize::MAX,
    })
}

fn build_linspace_f32(start: f32, end: f32, steps: usize) -> Vec<f32> {
    if steps == 1 {
        return vec![end];
    }
    let step = (end - start) / (steps as f32 - 1.0);
    let mut values = Vec::with_capacity(steps);
    for idx in 0..steps {
        if idx + 1 == steps {
            values.push(end);
        } else {
            values.push(start + step * idx as f32);
        }
    }
    values
}

fn build_linspace_f64(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 1 {
        return vec![end];
    }
    let step = (end - start) / (steps as f64 - 1.0);
    let mut values = Vec::with_capacity(steps);
    for idx in 0..steps {
        if idx + 1 == steps {
            values.push(end);
        } else {
            values.push(start + step * idx as f64);
        }
    }
    values
}

fn build_logspace_f32(start: f32, end: f32, steps: usize, base: f32) -> Result<Vec<f32>> {
    if !base.is_finite() || base <= 0.0 || base == 1.0 {
        return Err(Error::invalid_shape(
            "logspace base must be finite, > 0, and != 1",
        ));
    }
    let exponents = build_linspace_f32(start, end, steps);
    let mut values = Vec::with_capacity(steps);
    for exponent in exponents {
        values.push(base.powf(exponent));
    }
    Ok(values)
}

fn build_logspace_f64(start: f64, end: f64, steps: usize, base: f64) -> Result<Vec<f64>> {
    if !base.is_finite() || base <= 0.0 || base == 1.0 {
        return Err(Error::invalid_shape(
            "logspace base must be finite, > 0, and != 1",
        ));
    }
    let exponents = build_linspace_f64(start, end, steps);
    let mut values = Vec::with_capacity(steps);
    for exponent in exponents {
        values.push(base.powf(exponent));
    }
    Ok(values)
}
