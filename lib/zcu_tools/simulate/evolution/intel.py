"""Intel OpenCL complex128 batch matrix exponential; no CPU fallback."""

from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from .batch import BatchResult, PlateauFamily, batch_hamiltonians

# One independent work item per grid point. Scaling bounds ||A||_1 <= 1/4;
# 24 Taylor terms then binary squaring. No relaxed math or float intermediates.
_KERNEL = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
inline double2 mul(double2 a, double2 b) {
    return (double2)(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
}
inline void product(__private double2 *a, __private double2 *b,
                    __private double2 *out) {
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        double2 sum=(double2)(0.,0.);
        for (int k=0;k<N;k++) sum+=mul(a[i*N+k], b[k*N+j]);
        out[i*N+j]=sum;
    }
}
__kernel void exponential(__global const double2 *input,
                          __global double2 *output, double factor) {
    int node=get_global_id(0), offset=node*N*N;
    double2 a[N*N], term[N*N], result[N*N], temp[N*N];
    double norm=0.;
    for (int j=0;j<N;j++) {
        double sum=0.;
        for (int i=0;i<N;i++) {
            double2 h=input[offset+i*N+j];
            sum+=hypot(h.x,h.y)*fabs(factor);
        }
        norm=fmax(norm,sum);
    }
    int squares=norm>0.25 ? (int)ceil(log2(norm/0.25)) : 0;
    double scale=ldexp(1.,-squares);
    for (int i=0;i<N;i++) for (int j=0;j<N;j++) {
        int k=i*N+j; double2 h=input[offset+k];
        a[k]=(double2)(factor*h.y,-factor*h.x)*scale;
        term[k]=result[k]=(double2)(i==j ? 1. : 0.,0.);
    }
    for (int order=1;order<=24;order++) {
        product(term,a,temp);
        for (int k=0;k<N*N;k++) { term[k]=temp[k]/(double)order; result[k]+=term[k]; }
    }
    for (int s=0;s<squares;s++) {
        product(result,result,temp);
        for (int k=0;k<N*N;k++) result[k]=temp[k];
    }
    for (int k=0;k<N*N;k++) output[offset+k]=result[k];
}
"""


class IntelOpenCL:
    """Local Intel Arc FP64 batch implementation, selected explicitly.

    compiler_directory: explicit owned path for pyopencl compiler artifacts;
    never a global/default cache. No system/driver changes. Construction raises
    RuntimeError if Intel Arc or cl_khr_fp64 is unavailable. No CPU fallback.
    Context/queue live with this object; compute calls are synchronous and must
    not run concurrently on it. Runtime table lookup needs no OpenCL dependency.
    """

    def __init__(self, compiler_directory: Path) -> None:
        import pyopencl as cl

        devices = [
            device
            for platform in cl.get_platforms()
            for device in platform.get_devices()
            if "Intel" in device.vendor and "Arc" in device.name
        ]
        if not devices or "cl_khr_fp64" not in devices[0].extensions.split():
            raise RuntimeError("Intel Arc OpenCL with cl_khr_fp64 is required")
        compiler_directory.mkdir(parents=True, exist_ok=True)
        self.device = devices[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.compiler_directory = compiler_directory

    def compute(
        self, family: PlateauFamily, scales: NDArray[np.float64]
    ) -> BatchResult:
        """Batch exp(-2πi H(scale)*duration_ns), synchronous complex128 output.

        Raises OpenCL errors unchanged and RuntimeError on nonfinite output.
        Report compile/kernel/transfer times separately; device buffers/private
        arrays are build workspace, not runtime library resident cache.
        """
        import pyopencl as cl

        h = np.ascontiguousarray(batch_hamiltonians(family, scales))
        result = np.empty_like(h)
        n = h.shape[-1]
        start = perf_counter()
        program = cl.Program(self.context, _KERNEL).build(
            options=[f"-DN={n}"], cache_dir=str(self.compiler_directory)
        )
        compile_seconds = perf_counter() - start
        start = perf_counter()
        source = cl.Buffer(
            self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h
        )
        target = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, result.nbytes)
        self.queue.finish()
        transfer = perf_counter() - start
        start = perf_counter()
        program.exponential(
            self.queue,
            (len(h),),
            None,
            source,
            target,
            np.float64(2 * np.pi * family.duration_ns),
        ).wait()
        compute = perf_counter() - start
        start = perf_counter()
        cl.enqueue_copy(self.queue, result, target).wait()
        transfer += perf_counter() - start
        source.release()
        target.release()
        if not np.isfinite(result).all():
            raise RuntimeError("Intel batch exponential produced nonfinite values")
        return BatchResult(
            result,
            self.device.name,
            compute,
            transfer,
            compile_seconds,
            h.nbytes + result.nbytes,
            h.nbytes + result.nbytes + len(h) * 4 * n * n * 16,
        )
