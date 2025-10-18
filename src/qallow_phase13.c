#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>

void* harmonic_core(void* arg) {
    int id = *(int*)arg;
    printf("Thread %d active\n", id);
    return NULL;
}

int main() {
    const int n = 8;
    pthread_t threads[n];
    int ids[n];

    for (int i = 0; i < n; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, harmonic_core, &ids[i]);
    }

    for (int i = 0; i < n; i++) pthread_join(threads[i], NULL);
    printf("Qallow Phase 13 operational\n");
    return 0;
}
