/* bwhog: memory-bandwidth co-runner (contention type M).
 * Streams read+write over a buffer much larger than the last-level cache, so nearly every
 * access goes to DRAM, and prints achieved GB/s once per second (validation for gate G2).
 *   gcc -O2 -o bwhog bwhog.c
 *   ./bwhog <buffer_MB=256> <seconds=0 (forever)>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    size_t mb = argc > 1 ? (size_t)atol(argv[1]) : 256;
    double secs = argc > 2 ? atof(argv[2]) : 0;
    size_t n = mb * 1024 * 1024 / sizeof(long) / 2;
    long *a = malloc(n * sizeof(long)), *b = malloc(n * sizeof(long));
    if (!a || !b) { fprintf(stderr, "alloc failed\n"); return 1; }
    memset(a, 1, n * sizeof(long));
    memset(b, 2, n * sizeof(long));
    double t0 = now(), tl = t0, bytes = 0;
    long sink = 0;
    for (;;) {
        for (size_t i = 0; i < n; i++) { b[i] = a[i] + 1; }      /* 1 read + 1 write stream */
        sink += b[n / 2];
        bytes += 2.0 * n * sizeof(long);
        double t = now();
        if (t - tl >= 1.0) {
            printf("%.3f GB/s\n", bytes / (t - tl) / 1e9);
            fflush(stdout);
            bytes = 0; tl = t;
        }
        if (secs > 0 && t - t0 >= secs) break;
    }
    return (int)(sink & 1);
}
