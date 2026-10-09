/* cpuhog: compute-only co-runner (contention type P, the control).
 * Integer arithmetic in registers: keeps a core at ~100% with almost no DRAM traffic,
 * so it separates CPU-scheduling effects from memory-bandwidth effects.
 *   gcc -O2 -o cpuhog cpuhog.c
 *   ./cpuhog <seconds=0 (forever)>    prints millions of iterations per second
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    double secs = argc > 1 ? atof(argv[1]) : 0;
    volatile unsigned long x = 88172645463325252UL;
    double t0 = now(), tl = t0;
    unsigned long it = 0;
    for (;;) {
        for (int k = 0; k < 1000000; k++) { x ^= x << 13; x ^= x >> 7; x ^= x << 17; }
        it += 1000000;
        double t = now();
        if (t - tl >= 1.0) {
            printf("%.1f Mit/s\n", it / (t - tl) / 1e6);
            fflush(stdout);
            it = 0; tl = t;
        }
        if (secs > 0 && t - t0 >= secs) break;
    }
    return (int)(x & 1);
}
