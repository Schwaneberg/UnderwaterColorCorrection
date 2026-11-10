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

/* Match NumPy maximum/clip's propagation of NaN rather than fmax's masking. */
static float positive(float value) { return value < 0 ? 0 : value; }
static float clip01(float value) { return value < 0 ? 0 : value > 1 ? 1 : value; }

static float dot3(const float *v, const float *m) {
    return (v[0]*m[0]+v[1]*m[1])+v[2]*m[2];
}

static PyObject *manual_kernel(PyObject *args, int fused) {
    PyObject *a, *m, *p, *o;
    Py_buffer src={0}, matrix={0}, parameters={0}, dst={0};
    if (!PyArg_ParseTuple(args, "OOOO", &a, &m, &p, &o)) return NULL;
    if (!floats(a, &src, 0)) return NULL;
    if (!floats(m, &matrix, 0)) goto fail_manual;
    if (PyObject_GetBuffer(p, &parameters, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) goto fail_manual;
    if (PyObject_GetBuffer(o, &dst, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE) < 0) goto fail_manual;
    if (src.len%12 || matrix.len != (fused ? 30 : 21)*4 || parameters.len != 8*8 ||
        parameters.itemsize != 8 || !parameters.format || strcmp(parameters.format,"d") ||
        dst.len != src.len/4*(fused ? 2 : 8) || dst.itemsize != (fused ? 2 : 8) || !dst.format || strcmp(dst.format,fused ? "H" : "d") ||
        (uintptr_t)src.buf%4 || (uintptr_t)matrix.buf%4 || (uintptr_t)parameters.buf%8 || (uintptr_t)dst.buf%(fused ? 2 : 8) ||
        overlap(&src,&dst) || overlap(&matrix,&dst) || overlap(&parameters,&dst)) {
        PyErr_SetString(PyExc_ValueError,"Invalid manual color buffers, dimensions, alignment or overlap");
        goto fail_manual;
    }
    float transforms[30]; double controls[8];
    memcpy(transforms,matrix.buf,(size_t)matrix.len); memcpy(controls,parameters.buf,sizeof(controls));
    for (int i=0;i<(fused ? 30 : 21);i++) if (!isfinite(transforms[i])) {
        PyErr_SetString(PyExc_ValueError,"Manual color matrices must be finite"); goto fail_manual;
    }
    for (int i=0;i<8;i++) if (!isfinite(controls[i])) {
        PyErr_SetString(PyExc_ValueError,"Manual color parameters must be finite"); goto fail_manual;
    }
    const float *in=src.buf; double *out=dst.buf; uint16_t *codes=dst.buf;
    float exposure=(float)controls[0], contrast=(float)controls[1];
    float shadows=(float)controls[2], highlights=(float)controls[3], black=(float)controls[4];
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t at=0;at<src.len/4;at+=3) {
        float adapted[3], rgb[3], ap1[3], linear[3];
        const float *input=in+at;
        if (fused) {
            for (int c=0;c<3;c++) {
                float x=positive(input[c]);
                linear[c]=x<.081f ? x/4.5f : powf((x+.099f)/1.099f,1/.45f);
            }
            for (int c=0;c<3;c++) ap1[c]=dot3(linear,transforms+21+3*c);
            input=ap1;
        }
        for (int c=0;c<3;c++) adapted[c]=positive(dot3(input,transforms+3*c))*exposure;
        for (int c=0;c<3;c++) rgb[c]=dot3(adapted,transforms+9+3*c);
        float y=dot3(rgb,transforms+18); if (y<1e-7f) y=1e-7f;
        float base=y/.18f; if (base<1e-7f) base=1e-7f;
        float target=.18f*(contrast==1 ? base : powf(base,contrast));
        float shadow=clip01(1-y/.5f), high=clip01(y/.8f);
        float exponent=shadows*(shadow*shadow)+highlights*(high*high);
        target*=exponent==0 ? 1 : powf(2,exponent);
        target=positive(target-black);
        float factor=target/y;
        for (int c=0;c<3;c++) rgb[c]*=factor;
        y=dot3(rgb,transforms+18);
        float u=rgb[0]-rgb[1], v=rgb[2]-rgb[1];
        double u1=((double)u*controls[5]-(double)v*controls[6])*controls[7]/100;
        double v1=((double)u*controls[6]+(double)v*controls[5])*controls[7]/100;
        double green=(double)y-(double)transforms[18]*u1-(double)transforms[20]*v1;
        double finished[3]={green+u1,green,green+v1};
        if (!fused) {
            for (int c=0;c<3;c++) out[at+c]=finished[c];
        } else {
            /* Preserve both reference matrix round trips and float32 cast. */
            for (int c=0;c<3;c++) {
                const float *m=transforms+21+3*c;
                ap1[c]=(float)((finished[0]*m[0]+finished[1]*m[1])+finished[2]*m[2]);
            }
            for (int c=0;c<3;c++) rgb[c]=dot3(ap1,transforms+9+3*c);
            float luminance=bound(dot3(rgb,transforms+18),0,1), scale=1;
            for (int c=0;c<3;c++) {
                float d=rgb[c]-luminance;
                float room=d>0 ? (1-luminance)/fmaxf(d,1e-8f) : -luminance/fminf(d,-1e-8f);
                scale=fminf(scale,room);
            }
            for (int c=0;c<3;c++) {
                float v=bound(luminance+(rgb[c]-luminance)*scale,0,1);
                float encoded=bound(v<.018f ? v*4.5f : 1.099f*powf(v,.45f)-.099f,0,1);
                codes[at+c]=(uint16_t)nearbyintf(encoded*65535.f);
            }
        }
    }
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&src); PyBuffer_Release(&matrix); PyBuffer_Release(&parameters); PyBuffer_Release(&dst);
    Py_RETURN_NONE;
fail_manual:
    if (src.obj) PyBuffer_Release(&src);
    if (matrix.obj) PyBuffer_Release(&matrix);
    if (parameters.obj) PyBuffer_Release(&parameters);
    if (dst.obj) PyBuffer_Release(&dst);
    return NULL;
}

static PyObject *manual_color(PyObject *self, PyObject *args) { return manual_kernel(args,0); }
static PyObject *manual_rgb48(PyObject *self, PyObject *args) { return manual_kernel(args,1); }

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
    {"manual_color",manual_color,METH_VARARGS,"Fused manual pointwise color controls; float64 linear RGB output."},
    {"manual_rgb48",manual_rgb48,METH_VARARGS,"Fused Rec.709 decode, manual controls, encode and RGB48 quantization."},
    {"lut_rgb48",lut_rgb48,METH_VARARGS,"Fused uniform LUT, strength, and uint16 output (AVX2 by default when available)."},
    {"rgb48_backend",rgb48_backend,METH_NOARGS,"Return the CPU/OS-selected RGB48 kernel."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native(void) {
    rgb48_has_avx2 = detect_avx2();
    return PyModule_Create(&module);
}
