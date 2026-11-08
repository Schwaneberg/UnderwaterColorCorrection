#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include "_native_rgb48.h"

#if defined(UWCC_BUILD_AVX2)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

static int rgb48_has_avx2 = 0;

static int detect_avx2(void) {
#if defined(UWCC_BUILD_AVX2)
    unsigned int b, c;
#if defined(_MSC_VER)
    int regs[4];
    __cpuidex(regs, 0, 0);
    if (regs[0] < 7) return 0;
    __cpuidex(regs, 1, 0);
    c = (unsigned int)regs[2];
#else
    unsigned int a, d;
    if (__get_cpuid_max(0, NULL) < 7) return 0;
    __cpuid(1, a, b, c, d);
#endif
    /* XSAVE + OSXSAVE + AVX, then OS support for both XMM and YMM state. */
    if ((c & 0x1c000000u) != 0x1c000000u) return 0;
#if defined(_MSC_VER)
    if ((_xgetbv(0) & 6) != 6) return 0;
    __cpuidex(regs, 7, 0);
    b = (unsigned int)regs[1];
#else
    __asm__ volatile ("xgetbv" : "=a"(a), "=d"(d) : "c"(0));
    if ((a & 6) != 6) return 0;
    __cpuid_count(7, 0, a, b, c, d);
#endif
    return (b & (1u << 5)) != 0;
#else
    return 0;
#endif
}

/* Buffers remain pinned while the GIL is released. Never mutate the input. */
static int floats(PyObject *obj, Py_buffer *b, int writable) {
    if (PyObject_GetBuffer(obj, b, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS |
                          (writable ? PyBUF_WRITABLE : 0)) < 0) return 0;
    if (b->itemsize != 4 || !b->format || strcmp(b->format, "f") != 0) {
        PyBuffer_Release(b);
        PyErr_SetString(PyExc_ValueError, "Expected contiguous native float32 buffer");
        return 0;
    }
    return 1;
}
static float bound(float x, float lo, float hi) { return fminf(hi, fmaxf(lo, x)); }

static PyObject *lut_apply(PyObject *self, PyObject *args) {
    PyObject *a, *l, *o;
    Py_buffer src = {0}, lut = {0}, dst = {0};
    int n, uniform=0;
    if (!PyArg_ParseTuple(args, "OOOi|p", &a, &l, &o, &n, &uniform)) return NULL;
    if (!floats(a, &src, 0)) return NULL;
    if (!floats(l, &lut, 0)) goto fail;
    if (!floats(o, &dst, 1)) goto fail;
    if (n < 2 || n > 256 || src.len % 12 || dst.len != src.len ||
        lut.len != (Py_ssize_t)n*n*n*12 || dst.buf == src.buf || dst.buf == lut.buf) {
        PyErr_SetString(PyExc_ValueError, "Invalid LUT or RGB output dimensions"); goto fail;
    }
    const float *in = src.buf, *cube = lut.buf;
    float *out = dst.buf;
    /* Match the reference's double precision knots before casting fractions. */
    double grid[256];
    for (int i=0; i<n; i++) grid[i] = uniform ? (double)i/(n-1) : exp2((double)i/(n-1)*log2(17.0))-1;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t p=0; p<src.len/4; p+=3) {
        int idx[3], order[3]={0,1,2}; float f[3];
        for (int c=0; c<3; c++) {
            float coord=uniform ? bound(in[p+c],0,1)*(n-1) : bound(log2f(1+fmaxf(in[p+c],0))/log2f(17),0,1)*(n-1);
            idx[c]=(int)fminf(coord,n-2);
            f[c]=uniform ? bound(coord-idx[c],0,1) : bound((in[p+c]-grid[idx[c]])/(grid[idx[c]+1]-grid[idx[c]]),0,1);
        }
        /* Stable descending sort: equality keeps R,G,B ordering. */
        for (int i=1; i<3; i++) for (int j=i; j>0 && f[order[j]]>f[order[j-1]]; j--) {
            int t=order[j]; order[j]=order[j-1]; order[j-1]=t;
        }
        Py_ssize_t at=((Py_ssize_t)idx[0]*n*n+idx[1]*n+idx[2])*3;
        for (int c=0; c<3; c++) out[p+c]=cube[at+c]*(1-f[order[0]]);
        for (int i=0; i<3; i++) {
            idx[order[i]]++;
            at=((Py_ssize_t)idx[0]*n*n+idx[1]*n+idx[2])*3;
            float w=f[order[i]]-(i<2 ? f[order[i+1]] : 0);
            for (int c=0; c<3; c++) out[p+c]+=cube[at+c]*w;
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&src); PyBuffer_Release(&lut); PyBuffer_Release(&dst);
    Py_RETURN_NONE;
fail:
    if (src.obj) PyBuffer_Release(&src);
    if (lut.obj) PyBuffer_Release(&lut);
    if (dst.obj) PyBuffer_Release(&dst);
    return NULL;
}

static PyObject *encode709(PyObject *self, PyObject *args) {
    PyObject *a, *o; int tone;
    Py_buffer src={0}, dst={0};
    if (!PyArg_ParseTuple(args,"OOp",&a,&o,&tone)) return NULL;
    if (!floats(a,&src,0)) return NULL;
    if (!floats(o,&dst,1)) { PyBuffer_Release(&src); return NULL; }
    if (src.len%12 || dst.len!=src.len || dst.buf==src.buf) {
        PyErr_SetString(PyExc_ValueError,"Invalid RGB output dimensions");
        PyBuffer_Release(&src); PyBuffer_Release(&dst); return NULL;
    }
    const float *in=src.buf; float *out=dst.buf;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t p=0; p<src.len/4; p+=3) {
        float rgb[3]={in[p],in[p+1],in[p+2]};
        if (tone) {
            for (int c=0;c<3;c++) rgb[c]=fmaxf(rgb[c],0);
            float y=fmaxf(rgb[0]*.21263901f+rgb[1]*.71516868f+rgb[2]*.07219232f,1e-8f);
            float factor=(1+y/16)/(1+y)*1.6f;
            for (int c=0;c<3;c++) rgb[c]*=factor;
        }
        float y=bound(rgb[0]*.21263901f+rgb[1]*.71516868f+rgb[2]*.07219232f,0,1);
        float scale=1;
        for (int c=0;c<3;c++) {
            float d=rgb[c]-y;
            float room=d>0 ? (1-y)/fmaxf(d,1e-8f) : -y/fminf(d,-1e-8f);
            scale=fminf(scale,room);
        }
        for (int c=0;c<3;c++) {
            float v=bound(y+(rgb[c]-y)*scale,0,1);
            out[p+c]=bound(v<.018f ? v*4.5f : 1.099f*powf(v,.45f)-.099f,0,1);
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&src); PyBuffer_Release(&dst); Py_RETURN_NONE;
}
static float rgb48_unit(float value) {
    if (!(value >= 0)) return 0; /* Also sanitizes NaN, matching clip/nan_to_num. */
    return value > 1 ? 1 : value;
}

static int overlap(const Py_buffer *a, const Py_buffer *b) {
    uintptr_t x = (uintptr_t)a->buf, y = (uintptr_t)b->buf;
    if (!a->len || !b->len) return 0;
    return x <= y ? y-x < (uintptr_t)a->len : x-y < (uintptr_t)b->len;
}

static PyObject *lut_rgb48(PyObject *self, PyObject *args) {
    PyObject *a, *l, *o;
    Py_buffer src = {0}, lut = {0}, dst = {0};
    int n, use_avx2 = 1;
    double strength;
    if (!PyArg_ParseTuple(args, "OOOid|p", &a, &l, &o, &n, &strength, &use_avx2)) return NULL;
    if (!floats(a, &src, 0)) return NULL;
    if (!floats(l, &lut, 0)) goto fail_rgb48;
    if (PyObject_GetBuffer(o, &dst, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) goto fail_rgb48;
    if (n < 2 || n > 256 || src.len % 12 || dst.len != src.len/2 ||
        dst.itemsize != 2 || !dst.format || strcmp(dst.format, "H") ||
        lut.len != (Py_ssize_t)n*n*n*12 || !isfinite(strength) || strength < 0 || strength > 2 ||
        (uintptr_t)src.buf % 4 || (uintptr_t)lut.buf % 4 || (uintptr_t)dst.buf % 2 ||
        overlap(&src, &dst) || overlap(&lut, &dst)) {
        PyErr_SetString(PyExc_ValueError, "Invalid RGB48 buffers, LUT dimensions, alignment, overlap, or strength");
        goto fail_rgb48;
    }
    const float *in = src.buf, *cube = lut.buf;
    uint16_t *out = dst.buf;
    size_t values = (size_t)(src.len/4), start = 0;
    float blend = (float)strength;
    int strides[3] = {n*n*3, n*3, 3};
    int diagonal = strides[0]+strides[1]+strides[2];
    Py_BEGIN_ALLOW_THREADS
#if defined(UWCC_BUILD_AVX2)
    if (use_avx2 && rgb48_has_avx2)
        start = uwcc_lut_rgb48_avx2(in, cube, out, values, n, blend);
#endif
    /* Portable fused kernel also handles the AVX2 tail. Equality retains the
     * same stable R,G,B tetrahedron order as the original float LUT kernel. */
    for (size_t p = start; p < values; p += 3) {
        int idx[3]; float f[3];
        for (int c = 0; c < 3; c++) {
            float coord = rgb48_unit(in[p+c])*(n-1);
            idx[c] = (int)coord < n-2 ? (int)coord : n-2;
            f[c] = rgb48_unit(coord-idx[c]);
        }
        int first = f[0] >= f[1] ? (f[0] >= f[2] ? 0 : 2) : (f[1] >= f[2] ? 1 : 2);
        int last = f[2] <= f[1] ? (f[2] <= f[0] ? 2 : 0) : (f[1] <= f[0] ? 1 : 0);
        int middle = 3-first-last;
        int base = idx[0]*strides[0]+idx[1]*strides[1]+idx[2]*strides[2];
        int at[4] = {base, base+strides[first], base+diagonal-strides[last], base+diagonal};
        float weight[4] = {1-f[first], f[first]-f[middle], f[middle]-f[last], f[last]};
        for (int c = 0; c < 3; c++) {
            float value = cube[at[0]+c]*weight[0];
            for (int k = 1; k < 4; k++) value += cube[at[k]+c]*weight[k];
            float delta = value-in[p+c];
            float weighted = delta*blend;
            float adjusted = in[p+c]+weighted;
            float scaled = rgb48_unit(adjusted)*65535.0f;
            out[p+c] = (uint16_t)nearbyintf(scaled);
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&src); PyBuffer_Release(&lut); PyBuffer_Release(&dst);
    Py_RETURN_NONE;
fail_rgb48:
    if (src.obj) PyBuffer_Release(&src);
    if (lut.obj) PyBuffer_Release(&lut);
    if (dst.obj) PyBuffer_Release(&dst);
    return NULL;
}

static PyObject *rgb48_backend(PyObject *self, PyObject *unused) {
    return PyUnicode_FromString(rgb48_has_avx2 ? "avx2" : "scalar");
}

static PyMethodDef methods[]={
    {"apply_lut",lut_apply,METH_VARARGS,"Tetrahedral scene-linear LUT interpolation."},
    {"encode709",encode709,METH_VARARGS,"Fused SDR tone mapping, gamut compression and transfer."},
    {"lut_rgb48",lut_rgb48,METH_VARARGS,"Fused uniform LUT, strength, and uint16 output (AVX2 by default when available)."},
    {"rgb48_backend",rgb48_backend,METH_NOARGS,"Return the CPU/OS-selected RGB48 kernel."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native(void) {
    rgb48_has_avx2 = detect_avx2();
    return PyModule_Create(&module);
}
