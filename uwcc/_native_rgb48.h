#ifndef UWCC_NATIVE_RGB48_H
#define UWCC_NATIVE_RGB48_H

#include <stddef.h>
#include <stdint.h>

/* Returns the number of float channel values processed (eight RGB pixels per
 * block). The baseline translation unit handles any remaining pixels. */
size_t uwcc_lut_rgb48_avx2(const float *input, const float *cube, uint16_t *output,
                         size_t values, int n, float strength);

#endif
