#ifndef DL_INTEGRATION_H
#define DL_INTEGRATION_H

/* Deep Learning Integration Header */

#ifdef __cplusplus
extern "C" {
#endif

/* Placeholder for deep learning integration */
typedef struct {
    int version;
    char* backend;
} dl_backend_t;

/* Initialize DL backend */
dl_backend_t* dl_init(const char* backend_name);

/* Cleanup DL backend */
void dl_cleanup(dl_backend_t* backend);

/* Model support check */
int dl_model_supported(void);

/* Load model */
int dl_model_load(const char* model_path, int prefer_gpu);

/* Get last error */
const char* dl_model_last_error(void);

/* Unload model */
void dl_model_unload(void);

#ifdef __cplusplus
}
#endif

#endif /* DL_INTEGRATION_H */
