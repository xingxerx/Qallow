#include <math.h>
#include <string.h>
#include "qallow.h"

static double variance(const double* a, size_t n){
    double s=0, ss=0; for(size_t i=0;i<n;i++){ s+=a[i]; ss+=a[i]*a[i]; }
    double m=s/n; return (ss/n) - m*m;
}

void overlay_init(overlay_t* o, double seed){
    for (int i=0;i<QALLOW_NODES;i++){
        // simple deterministic seeding
        unsigned long long seed_ull = (unsigned long long)seed;
        unsigned long v = (unsigned long)(1469598103934665603ULL ^ (seed_ull + i*1099511628211ULL));
        o->nodes[i] = (double)((v >> 17) & 0x3FFFF) / (double)0x3FFFF; // ~[0,1]
    }
    o->stability = overlay_stability(o);
}

void overlay_apply_nudge(overlay_t* o, double delta){
    for (int i=0;i<QALLOW_NODES;i++){
        double x = o->nodes[i] + delta * (0.5 - o->nodes[i]); // pull toward 0.5
        if (x < 0) x = 0;
        if (x > 1) x = 1;
        o->nodes[i] = x;
    }
    o->stability = overlay_stability(o);
}

void overlay_propagate(overlay_t* src, overlay_t* dst, double factor){
    for (int i=0;i<QALLOW_NODES;i++){
        double ripple = (src->nodes[i] - 0.5) * factor;
        double x = dst->nodes[i] + ripple;
        if (x < 0) x = 0;
        if (x > 1) x = 1;
        dst->nodes[i] = x;
    }
    dst->stability = overlay_stability(dst);
}

double overlay_stability(overlay_t* o){
    // higher stability == lower variance
    double v = variance(o->nodes, QALLOW_NODES);
    return 1.0 / (1.0 + v);
}

