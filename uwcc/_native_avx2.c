/* AVX2-only translation unit. Never call before CPU/OS capability detection.
 * Keep multiply/add separate to preserve the scalar float32 operation order. */
#include <immintrin.h>
#include "_native_rgb48.h"

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
size_t uwcc_lut_rgb48_avx2(const float *in, const float *cube, uint16_t *out,
                         size_t values, int n, float blend) {
    int strides[3] = {n*n*3, n*3, 3};
    int diagonal = strides[0]+strides[1]+strides[2];
    size_t start=0;

    __m256 zero=_mm256_setzero_ps(),one=_mm256_set1_ps(1),all=_mm256_castsi256_ps(_mm256_set1_epi32(-1));
    __m256i input_offsets=_mm256_setr_epi32(0,3,6,9,12,15,18,21);
    for(;values-start>=24;start+=24){
        __m256 source[3],f8[3];__m256i ix[3];
        for(int c=0;c<3;c++){
            source[c]=_mm256_i32gather_ps(in+start+c,input_offsets,4);
            __m256 coord=_mm256_mul_ps(_mm256_min_ps(_mm256_max_ps(source[c],zero),one),_mm256_set1_ps((float)(n-1)));
            ix[c]=_mm256_min_epi32(_mm256_cvttps_epi32(coord),_mm256_set1_epi32(n-2));
            f8[c]=_mm256_min_ps(_mm256_max_ps(_mm256_sub_ps(coord,_mm256_cvtepi32_ps(ix[c])),zero),one);
        }
        __m256 first0=_mm256_and_ps(_mm256_cmp_ps(f8[0],f8[1],_CMP_GE_OQ),_mm256_cmp_ps(f8[0],f8[2],_CMP_GE_OQ));
        __m256 first1=_mm256_and_ps(_mm256_cmp_ps(f8[1],f8[0],_CMP_GT_OQ),_mm256_cmp_ps(f8[1],f8[2],_CMP_GE_OQ));
        __m256 last0=_mm256_and_ps(_mm256_cmp_ps(f8[0],f8[1],_CMP_LT_OQ),_mm256_cmp_ps(f8[0],f8[2],_CMP_LT_OQ));
        __m256 last1=_mm256_and_ps(_mm256_cmp_ps(f8[1],f8[0],_CMP_LE_OQ),_mm256_cmp_ps(f8[1],f8[2],_CMP_LT_OQ));
        __m256 middle0=_mm256_andnot_ps(_mm256_or_ps(first0,last0),all);
        __m256 middle1=_mm256_andnot_ps(_mm256_or_ps(first1,last1),all);
        __m256 high=_mm256_max_ps(_mm256_max_ps(f8[0],f8[1]),f8[2]);
        __m256 low=_mm256_min_ps(_mm256_min_ps(f8[0],f8[1]),f8[2]);
        __m256 middle=_mm256_blendv_ps(_mm256_blendv_ps(f8[2],f8[1],middle1),f8[0],middle0);
        __m256i firstoff=_mm256_blendv_epi8(_mm256_blendv_epi8(_mm256_set1_epi32(strides[2]),_mm256_set1_epi32(strides[1]),_mm256_castps_si256(first1)),_mm256_set1_epi32(strides[0]),_mm256_castps_si256(first0));
        __m256i lastoff=_mm256_blendv_epi8(_mm256_blendv_epi8(_mm256_set1_epi32(strides[2]),_mm256_set1_epi32(strides[1]),_mm256_castps_si256(last1)),_mm256_set1_epi32(strides[0]),_mm256_castps_si256(last0));
        __m256i base=_mm256_add_epi32(_mm256_add_epi32(_mm256_mullo_epi32(ix[0],_mm256_set1_epi32(strides[0])),_mm256_mullo_epi32(ix[1],_mm256_set1_epi32(strides[1]))),_mm256_mullo_epi32(ix[2],_mm256_set1_epi32(3)));
        __m256i offsets[4]={base,_mm256_add_epi32(base,firstoff),_mm256_sub_epi32(_mm256_add_epi32(base,_mm256_set1_epi32(diagonal)),lastoff),_mm256_add_epi32(base,_mm256_set1_epi32(diagonal))};
        __m256 weights[4]={_mm256_sub_ps(one,high),_mm256_sub_ps(high,middle),_mm256_sub_ps(middle,low),low};
        for(int c=0;c<3;c++){
            __m256 value=_mm256_mul_ps(_mm256_i32gather_ps(cube+c,offsets[0],4),weights[0]);
            for(int k=1;k<4;k++)value=_mm256_add_ps(value,_mm256_mul_ps(_mm256_i32gather_ps(cube+c,offsets[k],4),weights[k]));
            value=_mm256_add_ps(source[c],_mm256_mul_ps(_mm256_sub_ps(value,source[c]),_mm256_set1_ps(blend)));
            value=_mm256_min_ps(_mm256_max_ps(value,zero),one);
            __m256i rounded=_mm256_cvtps_epi32(_mm256_mul_ps(value,_mm256_set1_ps(65535)));
            int32_t packed[8];_mm256_storeu_si256((__m256i*)packed,rounded);
            for(int i=0;i<8;i++)out[start+3*i+c]=(uint16_t)packed[i];
        }
    }
    _mm256_zeroupper();
    return start;
}
