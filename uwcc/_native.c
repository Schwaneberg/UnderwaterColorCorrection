#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

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
static PyMethodDef methods[]={
    {"apply_lut",lut_apply,METH_VARARGS,"Tetrahedral scene-linear LUT interpolation."},
    {"encode709",encode709,METH_VARARGS,"Fused SDR tone mapping, gamut compression and transfer."},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef module={PyModuleDef_HEAD_INIT,"_native",NULL,-1,methods};
PyMODINIT_FUNC PyInit__native(void) { return PyModule_Create(&module); }
