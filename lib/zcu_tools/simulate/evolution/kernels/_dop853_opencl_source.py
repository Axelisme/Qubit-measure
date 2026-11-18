"""OpenCL translation of the dop853 family's interaction-frame endpoint solver."""

from ._dop853 import (
    DP54_ERROR_EXPONENT,
    ENDPOINT_ATTEMPT_LIMIT,
    ERROR_NORM_BLEND,
    ROW_INITIAL_STEP_NS,
    STEP_ERROR_EXPONENT,
    STEP_MAX_FACTOR,
    STEP_MIN_FACTOR,
    STEP_SAFETY,
    endpoint_coefficients,
)


def opencl_source() -> str:
    """Return the common source for both tableau variants.

    Compile with D (density dimension), WG (power-of-two reduction width) and
    DP54 (1 for DP5(4), 0 for DOP853). The state stays in the interaction frame
    across bounded launches.
    """
    definitions = [
        "#pragma OPENCL EXTENSION cl_khr_fp64 : enable",
        "#if DP54",
        *_tableau_definitions(dp54=True),
        "#else",
        *_tableau_definitions(dp54=False),
        "#endif",
    ]
    for name, scalar in {
        "SEED": ROW_INITIAL_STEP_NS,
        "SAFETY": STEP_SAFETY,
        "MIN_FACTOR": STEP_MIN_FACTOR,
        "MAX_FACTOR": STEP_MAX_FACTOR,
        "NORM_BLEND": ERROR_NORM_BLEND,
    }.items():
        definitions.append(f"#define {name} ({scalar:.17e}f)")
    definitions.append(f"#define ATTEMPT_LIMIT {ENDPOINT_ATTEMPT_LIMIT}")
    return "\n".join([*definitions, _KERNEL])


def _tableau_definitions(*, dp54: bool) -> list[str]:
    coefficients = endpoint_coefficients(dp54=dp54)
    exponent = DP54_ERROR_EXPONENT if dp54 else STEP_ERROR_EXPONENT
    definitions = [
        f"#define STAGES {len(coefficients['C'])}",
        f"#define EXPONENT ({exponent:.17e}f)",
    ]
    for name, values in coefficients.items():
        shape = "".join(f"[{axis}]" for axis in values.shape)
        numbers = [f"{v:.17e}f" for v in values.ravel()]
        if values.ndim == 2:
            width = values.shape[1]
            numbers = [
                "{" + ",".join(numbers[i : i + width]) + "}"
                for i in range(0, len(numbers), width)
            ]
        definitions.append(f"__constant float {name}{shape} = {{{','.join(numbers)}}};")
    return definitions


_KERNEL = r"""
#define Q (D * D)
#define TAU 6.283185307179586476925286766559
#define FENCE barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

// Local memory belongs to one work-group, which advances one timeline.
typedef struct {
    float2 rho[Q];
    float2 h[Q];
    float2 product[Q];
    float2 energy_origin[D];
    float2 energy_phase[D];
    float2 carrier_origin[2];
    float2 drive[2];
    float2 norms[WG];
} Scratch;

inline float2 mul(float2 a, float2 b) {
    return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}
inline float2 conj2(float2 a) { return (float2)(a.x, -a.y); }
inline float2 phasor(float angle) {
    float cosine;
    float sine = sincos(angle, &cosine);
    return (float2)(cosine, sine);
}
inline float2 origin_phasor(double angle) {
    double cosine;
    double sine = sincos(angle, &cosine);
    return (float2)((float)cosine, (float)sine);
}

// Only the step origin needs double-precision argument reduction. Each stage
// composes that phasor with a float rotation over its small step-local offset.
inline void step_origin(__local Scratch *work, __constant const double *energies,
                       double4 tone, double time) {
    int lane = get_local_id(0);
    for (int i = lane; i < D; i += WG)
        work->energy_origin[i] = origin_phasor(TAU * energies[i] * time);
    if (lane < 2)
        work->carrier_origin[lane] = origin_phasor(TAU * tone[lane] * time + tone[lane+2]);
    FENCE;
}

inline void derivative(__local Scratch *work, __global float2 *out,
                       __constant const float2 *operators, __constant const float *bath) {
    int lane = get_local_id(0);
    for (int q = lane; q < Q; q += WG) {
        float2 h = (float2)(0);
        for (int tone = 0; tone < 2; ++tone) {
            float2 z = work->drive[tone];
            h += z.x * operators[(2*tone)*Q + q] + z.y * operators[(2*tone+1)*Q + q];
        }
        work->h[q] = mul(h, mul(work->energy_phase[q/D], conj2(work->energy_phase[q%D])));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = lane; q < Q; q += WG) {
        int i = q/D, j = q%D;
        float2 product = (float2)(0);
        for (int k = 0; k < D; ++k)
            product += mul(work->h[i*D+k], work->rho[k*D+j]);
        work->product[q] = product;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int q = lane; q < Q; q += WG) {
        int i = q/D, j = q%D;
        float2 comm = work->product[q] - conj2(work->product[j*D+i]);
        float2 value = (float)TAU * (float2)(comm.y, -comm.x) - bath[Q+q] * work->rho[q];
        if (i == j)
            for (int k = 0; k < D; ++k)
                value += bath[i*D+k] * work->rho[k*D+k];
        out[q] = value;
    }
    FENCE;
}

inline void stage(__local Scratch *work, __global const float2 *env,
                  __constant const double *energies, double4 tone,
                  float offset, float fraction) {
    int lane = get_local_id(0);
    for (int i = lane; i < D; i += WG)
        work->energy_phase[i] = mul(work->energy_origin[i], phasor((float)TAU * (float)energies[i] * offset));
    if (lane < 2) {
        float2 sample = mix(env[lane], env[lane+2], fraction);
        work->drive[lane] = mul(sample, mul(work->carrier_origin[lane], phasor((float)TAU * (float)tone[lane] * offset)));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

inline float embedded_error(__local Scratch *work, __global const float2 *state,
                            __global const float2 *slopes, float step, float2 tolerance) {
    int lane = get_local_id(0);
    float2 norms = (float2)(0);
    for (int q = lane; q < Q; q += WG) {
        float2 value = state[q], e5 = (float2)(0);
#if !DP54
        float2 e3 = (float2)(0);
#endif
        for (int s = 0; s < STAGES; ++s) {
            float2 slope = slopes[s*Q+q];
            if (B[s] != 0) value += (step * B[s]) * slope;
            e5 += E5[s] * slope;
#if !DP54
            e3 += E3[s] * slope;
#endif
        }
        work->rho[q] = value;
        float scale = tolerance.y + tolerance.x * fmax(length(state[q]), length(value));
        e5 /= scale;
#if DP54
        norms += (float2)(dot(e5, e5), 0.0f);
#else
        e3 /= scale;
        norms += (float2)(dot(e5, e5), dot(e3, e3));
#endif
        if (!all(isfinite(value))) norms.x = INFINITY;
    }
    work->norms[lane] = norms;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int stride = WG/2; stride > 0; stride >>= 1) {
        if (lane < stride) work->norms[lane] += work->norms[lane+stride];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float2 total = work->norms[0];
#if DP54
    return step * sqrt(total.x / Q);
#else
    return all(total == 0) ? 0.0f : step * total.x / sqrt((total.x + NORM_BLEND * total.y) * Q);
#endif
}

// progress = (interval, trials since knot, previous rejection, error code).
// Only progress, clocks and the state survive a dispatch; stage work is local.
__kernel void advance(__global float2 *states, __global float2 *derivatives,
                      __global int4 *progress, __global double *elapsed,
                      __global float *steps, __constant const double *energies,
                      __constant const float2 *operators, __constant const float *bath,
                      __global const double *knots, __global const float2 *samples,
                      __global const double4 *tones, __global const int *interval_counts,
                      int width, float rtol, float atol, int trial_limit,
                      __global const int *active_rows) {
    int row = active_rows[get_group_id(0)], lane = get_local_id(0);
    __global float2 *state = states + row*Q;
    __global float2 *slopes = derivatives + row*STAGES*Q;
    __global const double *times = knots + row*width;
    __global const float2 *env = samples + 2*row*width;
    __local Scratch work;
    __local int4 cursor;
    __local double now;
    __local float step;
    __local int take;
    if (lane == 0) {
        cursor = progress[row];
        now = elapsed[row];
        step = steps[row];
    }
    FENCE;
    for (int attempt = 0; attempt < trial_limit; ++attempt) {
        // Snapshot shared controller fields before lane zero can advance them.
        int4 pos = cursor;
        double clock = now;
        float nominal = step;
        FENCE;
        if (pos.w || pos.x == interval_counts[row]) break;
        double left = times[pos.x], span = times[pos.x+1] - left;
        double remaining = span - clock;
        float h = (float)fmin((double)nominal, remaining);
        if (h <= 0 || clock + (double)h == clock || pos.y == ATTEMPT_LIMIT) {
            if (lane == 0) cursor.w = 1;
            break;
        }
        double4 tone = tones[row];
        step_origin(&work, energies, tone, left + clock);
        for (int s = 0; s < STAGES; ++s) {
            for (int q = lane; q < Q; q += WG) {
                float2 value = state[q];
                for (int p = 0; p < s; ++p)
                    if (A[s][p] != 0) value += (h * A[s][p]) * slopes[p*Q+q];
                work.rho[q] = value;
            }
            float offset = C[s] * h;
            float fraction = clamp((float)(clock / span) + offset / (float)span, 0.0f, 1.0f);
            stage(&work, env + 2*pos.x, energies, tone, offset, fraction);
            derivative(&work, slopes + s*Q, operators, bath);
        }
        float error = embedded_error(&work, state, slopes, h, (float2)(rtol, atol));
        if (lane == 0) {
            ++cursor.y;
            take = isfinite(error) && error <= 1;
            if (!isfinite(error)) cursor.w = 2;
            else {
                float factor = error == 0 ? MAX_FACTOR : SAFETY * pow(error, EXPONENT);
                if (take && (double)nominal >= remaining) {
                    ++cursor.x;
                    cursor.y = cursor.z = 0;
                    now = 0;
                    step = SEED;
                } else if (take) {
                    now += (double)h;
                    step = h * fmin(pos.z ? 1.0f : MAX_FACTOR, factor);
                    cursor.z = 0;
                } else {
                    step = h * fmax(MIN_FACTOR, factor);
                    cursor.z = 1;
                }
            }
        }
        FENCE;
        if (take)
            for (int q = lane; q < Q; q += WG) state[q] = work.rho[q];
        FENCE;
    }
    if (lane == 0) {
        progress[row] = cursor;
        elapsed[row] = now;
        steps[row] = step;
    }
}
"""
