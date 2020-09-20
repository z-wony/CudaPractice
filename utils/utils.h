#ifndef __UTIL_UTILS_H__
#define __UTIL_UTILS_H__

#ifdef __cplusplus
extern "C" {
#endif

void write_JPEG_file(const char * filename, int quality, unsigned char *rgb_buffer, int image_width, int image_height);

/*
 * @remarks buffer_rgb must be released by free
 */
int read_JPEG_file(const char *filename, unsigned char **buffer_rgb, int *width, int *height);

#ifdef __cplusplus
}
#endif

#endif /* __UTIL_UTILS_H__ */
