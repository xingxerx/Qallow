#include "qallow/profiling.h"

#include "qallow/logging.h"

#include <chrono>

#if defined(__has_include)
#  if __has_include(<nvToolsExt.h>)
#    define QALLOW_HAVE_NVTX 1
#    include <nvToolsExt.h>
#  else
#    define QALLOW_HAVE_NVTX 0
#  endif
#else
#  define QALLOW_HAVE_NVTX 0
#endif

namespace {
double now_ns() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<double>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
}
}  // namespace

extern "C" {

qallow_profile_scope_t qallow_profile_scope_enter(const char* name) {
    qallow_profile_scope_t scope;
    scope.name = name ? name : "scope";
    scope.start_ns = now_ns();
    scope.active = 1;
#if QALLOW_HAVE_NVTX
    nvtxRangePushA(scope.name);
#endif
    return scope;
}

void qallow_profile_scope_exit(qallow_profile_scope_t* scope) {
    if (!scope || !scope->active) {
        return;
    }

    double elapsed_ms = (now_ns() - scope->start_ns) / 1.0e6;
    qallow_log_info("profile", "%s elapsed_ms=%.3f", scope->name, elapsed_ms);

#if QALLOW_HAVE_NVTX
    nvtxRangePop();
#endif
    scope->active = 0;
}

}  // extern "C"
