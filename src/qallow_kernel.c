#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "qallow.h"

static int ethics_pass(qvm_t* vm){
    double E = vm->ethics_S + vm->ethics_C + vm->ethics_H;
    return E >= 1.0;
}

void qallow_init(qvm_t* vm){
    vm->tick = 0;
    vm->ethics_S = 0.34; vm->ethics_C = 0.33; vm->ethics_H = 0.34;
    vm->decoherence = 0.0004;
    overlay_init(&vm->orbital,  12345);
    overlay_init(&vm->river,    67890);
    overlay_init(&vm->mycelial, 424242);
}

double qallow_global_stability(qvm_t* vm){
    return (vm->orbital.stability + vm->river.stability + vm->mycelial.stability)/3.0;
}

void qallow_tick(qvm_t* vm, unsigned long seed){
    // 1) GPU: photonic probabilistic fill (host buffer reused: orbital.nodes)
    runPhotonicSimulation(vm->orbital.nodes, QALLOW_NODES, seed);

    // 2) CPU nudge toward midline (soften extremes),
    overlay_apply_nudge(&vm->orbital, 0.10);

    // 3) ripple Orbital -> River
    overlay_propagate(&vm->orbital, &vm->river, 0.05);

    // 4) GPU: quantum-inspired optimizer on Mycelial
    runQuantumOptimizer(vm->mycelial.nodes, QALLOW_NODES);

    // 5) ripple River -> Mycelial (small)
    overlay_propagate(&vm->river, &vm->mycelial, 0.02);

    // 6) ethics & safeguard
    vm->decoherence *= 0.999; // decay
    if (!ethics_pass(vm) || vm->decoherence > 0.0015){
        // simple pause action: pull all toward neutral
        overlay_apply_nudge(&vm->orbital,  0.02);
        overlay_apply_nudge(&vm->river,    0.02);
        overlay_apply_nudge(&vm->mycelial, 0.02);
    }

    vm->tick++;
}

int main(void){
    qvm_t vm;
    qallow_init(&vm);

    printf("[Qallow] Native start. Nodes=%d\n", QALLOW_NODES);
    for (int s=0; s<60; ++s){
        unsigned long seed = (unsigned long)time(NULL) ^ (unsigned long)(s*1315423911u);
        qallow_tick(&vm, seed);
        printf("[%02d] Orbital=%.4f River=%.4f Mycelial=%.4f | Global=%.4f | Deco=%.5f\n",
            s+1, vm.orbital.stability, vm.river.stability, vm.mycelial.stability,
            qallow_global_stability(&vm), vm.decoherence);
        fflush(stdout);
    }
    puts("[Qallow] Done.");
    return 0;
}

