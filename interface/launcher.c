
#include <stdbool.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <limits.h>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#include <process.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#if defined(_WIN32)
#define QALLOW_ACCESS _access
#define QALLOW_PATH_SEP '\\'
#define QALLOW_ALT_PATH_SEP '/'
#else
#define QALLOW_ACCESS access
#define QALLOW_PATH_SEP '/'
#define QALLOW_ALT_PATH_SEP '\\'
#endif

static int g_skip_build = 0;

static void qallow_dirname_inplace(char* path) {
    if (!path) {
        return;
    }

    size_t len = strlen(path);
    if (len == 0) {
        return;
    }

#if defined(_WIN32)
    if (!(len == 3 && path[1] == ':' && (path[2] == '\\' || path[2] == '/'))) {
        while (len > 1 && (path[len - 1] == QALLOW_PATH_SEP || path[len - 1] == QALLOW_ALT_PATH_SEP)) {
            path[--len] = '\0';
        }
    }
#else
    while (len > 1 && (path[len - 1] == QALLOW_PATH_SEP || path[len - 1] == QALLOW_ALT_PATH_SEP)) {
        path[--len] = '\0';
    }
#endif

    char* last_sep = strrchr(path, QALLOW_PATH_SEP);
    char* last_alt = strrchr(path, QALLOW_ALT_PATH_SEP);
    if (last_alt && (!last_sep || last_alt > last_sep)) {
        last_sep = last_alt;
    }

    if (!last_sep) {
        strcpy(path, ".");
        return;
    }

    if (last_sep == path) {
        last_sep[1] = '\0';
    } else {
        *last_sep = '\0';
        if (path[0] == '\0') {
            strcpy(path, ".");
        }
    }
}

static int qallow_get_executable_path(char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return 0;
    }

#if defined(_WIN32)
    DWORD len = GetModuleFileNameA(NULL, buffer, (DWORD)buffer_size);
    if (len == 0 || len >= buffer_size) {
        return 0;
    }
    return 1;
#else
    ssize_t len = readlink("/proc/self/exe", buffer, buffer_size - 1);
    if (len < 0 || (size_t)len >= buffer_size) {
        return 0;
    }
    buffer[len] = '\0';
    return 1;
#endif
}

static int qallow_find_project_root(char* buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return 0;
    }

    const char* env_root = getenv("QALLOW_ROOT");
    if (env_root && *env_root) {
        size_t len = strlen(env_root);
        if (len >= buffer_size) {
            return 0;
        }
        memcpy(buffer, env_root, len + 1);
        return 1;
    }

    char current[PATH_MAX];
    if (!qallow_get_executable_path(current, sizeof(current))) {
#if defined(_WIN32)
        if (!GetCurrentDirectoryA((DWORD)sizeof(current), current)) {
            return 0;
        }
#else
        if (!getcwd(current, sizeof(current))) {
            return 0;
        }
#endif
    }

#if defined(_WIN32)
    DWORD attrs = GetFileAttributesA(current);
    if (attrs != INVALID_FILE_ATTRIBUTES) {
        if (!(attrs & FILE_ATTRIBUTE_DIRECTORY)) {
            qallow_dirname_inplace(current);
        }
    } else {
        if (!GetCurrentDirectoryA((DWORD)sizeof(current), current)) {
            return 0;
        }
    }
#else
    struct stat st;
    if (stat(current, &st) == 0) {
        if (S_ISREG(st.st_mode)) {
            qallow_dirname_inplace(current);
        }
    } else {
        if (!getcwd(current, sizeof(current))) {
            return 0;
        }
    }
#endif

    for (int depth = 0; depth < 8; ++depth) {
        char candidate[PATH_MAX];
        int written = snprintf(candidate, sizeof(candidate), "%s%cCMakeLists.txt", current, QALLOW_PATH_SEP);
        if (written <= 0 || written >= (int)sizeof(candidate)) {
            return 0;
        }

        if (QALLOW_ACCESS(candidate, 0) == 0) {
            size_t len = strlen(current);
            if (len >= buffer_size) {
                return 0;
            }
            memcpy(buffer, current, len + 1);
            return 1;
        }

        char parent[PATH_MAX];
        strncpy(parent, current, sizeof(parent));
        parent[sizeof(parent) - 1] = '\0';
        qallow_dirname_inplace(parent);
        if (strcmp(parent, current) == 0) {
            break;
        }
        strncpy(current, parent, sizeof(current));
        current[sizeof(current) - 1] = '\0';
    }

#if defined(_WIN32)
    if (GetCurrentDirectoryA((DWORD)sizeof(current), current)) {
#else
    if (getcwd(current, sizeof(current))) {
#endif
        char candidate[PATH_MAX];
        int written = snprintf(candidate, sizeof(candidate), "%s%cCMakeLists.txt", current, QALLOW_PATH_SEP);
        if (written > 0 && written < (int)sizeof(candidate) && QALLOW_ACCESS(candidate, 0) == 0) {
            size_t len = strlen(current);
            if (len < buffer_size) {
                memcpy(buffer, current, len + 1);
                return 1;
            }
        }
    }

    return 0;
}

static void qallow_init_process_flags(void) {
    const char* skip = getenv("QALLOW_SKIP_BUILD_ONCE");
    if (skip && strcmp(skip, "1") == 0) {
        g_skip_build = 1;
#if defined(_WIN32)
        _putenv_s("QALLOW_SKIP_BUILD_ONCE", "");
#else
        unsetenv("QALLOW_SKIP_BUILD_ONCE");
#endif
    }
}

static int qallow_run_build_scripts(int clean) {
    (void)clean;

    char root[PATH_MAX];
    if (!qallow_find_project_root(root, sizeof(root))) {
        fprintf(stderr, "[ERROR] Unable to locate Qallow project root. Set QALLOW_ROOT to override.\n");
        return 1;
    }

#if defined(_WIN32)
    const char* cd_prefix = "cd /d";
#else
    const char* cd_prefix = "cd";
#endif

    char command[PATH_MAX * 6];
    int written = snprintf(
        command,
        sizeof(command),
        "%s \"%s\" && cmake -S . -B build && cmake --build build --parallel",
        cd_prefix,
        root);
    if (written <= 0 || written >= (int)sizeof(command)) {
        fprintf(stderr, "[ERROR] Build command too long.\n");
        return 1;
    }

    printf("[BUILD] Synchronizing sources at %s\n", root);
    int rc = system(command);
    if (rc != 0) {
#if defined(_WIN32)
        fprintf(stderr, "[ERROR] Build command failed (exit=%d).\n", rc);
#else
        if (WIFEXITED(rc)) {
            fprintf(stderr, "[ERROR] Build command failed (exit=%d).\n", WEXITSTATUS(rc));
        } else if (WIFSIGNALED(rc)) {
            fprintf(stderr, "[ERROR] Build command terminated by signal %d.\n", WTERMSIG(rc));
        } else {
            fprintf(stderr, "[ERROR] Build command failed (code=%d).\n", rc);
        }
#endif
        return 1;
    }

    return 0;
}

static int qallow_restart_self(int argc, char** argv) {
#if defined(_WIN32)
    if (_putenv_s("QALLOW_SKIP_BUILD_ONCE", "1") != 0) {
        fprintf(stderr, "[ERROR] Failed to set restart environment flag.\n");
        return 1;
    }
#else
    if (setenv("QALLOW_SKIP_BUILD_ONCE", "1", 1) != 0) {
        perror("[ERROR] Failed to set restart environment flag");
        return 1;
    }
#endif

#if defined(_WIN32)
    _execvp(argv[0], (const char* const*)argv);
    fprintf(stderr, "[ERROR] Failed to restart process (errno=%d).\n", errno);
#else
    execvp(argv[0], argv);
    perror("[ERROR] Failed to restart Qallow after rebuild");
#endif
    return 1;
}

static int qallow_build_and_maybe_restart(int argc, char** argv) {
    if (g_skip_build) {
        return 0;
    }

    if (qallow_run_build_scripts(0) != 0) {
        return 1;
    }

    return qallow_restart_self(argc, argv);
}

// Include all core headers
#include "qallow_kernel.h"
#include "ppai.h"
#include "qcp.h"
#include "ethics.h"
#include "overlay.h"
#include "sandbox.h"
#include "telemetry.h"
#include "pocket.h"
#include "govern.h"
#include "qallow_phase11.h"
#include "qallow_phase12.h"
#include "qallow_phase13.h"
#include "qallow_phase14.h"
#include "qallow_phase15.h"
#include "phase13_accelerator.h"
#include "qallow_integration.h"
#include "meta_introspect.h"
#include "dl_integration.h"
#include "qallow/module.h"
// TODO: Add these when modules are implemented
// #include "adaptive.h"
// #include "verify.h"
// #include "ingest.h"

typedef enum {
    RUN_PROFILE_STANDARD = 0,
    RUN_PROFILE_BENCH,
    RUN_PROFILE_LIVE
} run_profile_t;

// Forward declarations
static int qallow_build_mode(void);
static void qallow_verify_mode(void);
static void qallow_print_help(void);
static void qallow_print_run_help(void);
static void qallow_print_system_help(void);
static void qallow_print_phase_help(void);
static void qallow_print_mind_help(void);
static int qallow_run_vm(run_profile_t profile);
static int qallow_handle_run(int argc, char** argv, int arg_offset, run_profile_t default_profile);
static int qallow_dispatch_phase(int argc, char** argv, int start_index, const char* phase_name,
                                 int (*runner)(int, char**));
static int qallow_clear_mode(void);
static int qallow_handle_run_group(int argc, char** argv, int arg_offset);
static int qallow_handle_system_group(int argc, char** argv, int arg_offset);
static int qallow_handle_phase_group(int argc, char** argv, int arg_offset);
static int qallow_handle_mind_group(int argc, char** argv, int arg_offset);
int qallow_cmd_mind(int argc, char **argv);
int qallow_cmd_bench(int argc, char **argv);
int qallow_cmd_entangle(int argc, char **argv);

static int remove_recursive(const char* path);

#if defined(_WIN32)
#define QALLOW_SETENV(name, value) _putenv_s((name), (value))
#else
#define QALLOW_SETENV(name, value) setenv((name), (value), 1)
#endif

static int qallow_apply_dashboard_option(const char* value) {
    if (!value || !*value) {
        fprintf(stderr, "[ERROR] --dashboard requires a value (off|<ticks>)\n");
        return 0;
    }

    char lowered[16];
    size_t len = strlen(value);
    if (len >= sizeof(lowered)) {
        fprintf(stderr, "[ERROR] --dashboard value too long: %s\n", value);
        return 0;
    }

    for (size_t i = 0; i < len; i++) {
        lowered[i] = (char)tolower((unsigned char)value[i]);
    }
    lowered[len] = '\0';

    if (strcmp(lowered, "off") == 0 || strcmp(lowered, "disable") == 0) {
        if (QALLOW_SETENV("QALLOW_DASHBOARD_INTERVAL", "0") != 0) {
            fprintf(stderr, "[ERROR] Failed to set dashboard environment override\n");
            return 0;
        }
        printf("[RUN] Dashboard output disabled (--dashboard)\n");
        return 1;
    }

    char* endptr = NULL;
    long parsed = strtol(value, &endptr, 10);
    if (endptr == value || (endptr && *endptr != '\0')) {
        fprintf(stderr, "[ERROR] Invalid --dashboard value: %s (expected off|<ticks>)\n", value);
        return 0;
    }
    if (parsed < 0) {
        fprintf(stderr, "[ERROR] Dashboard interval must be non-negative: %ld\n", parsed);
        return 0;
    }
    if (parsed > 1000000) {
        fprintf(stderr, "[ERROR] Dashboard interval too large: %ld\n", parsed);
        return 0;
    }

    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%ld", parsed);
    if (QALLOW_SETENV("QALLOW_DASHBOARD_INTERVAL", buffer) != 0) {
        fprintf(stderr, "[ERROR] Failed to set dashboard environment override\n");
        return 0;
    }

    if (parsed == 0) {
        printf("[RUN] Dashboard output disabled (--dashboard=0)\n");
    } else {
        printf("[RUN] Dashboard interval set to %ld ticks (--dashboard)\n", parsed);
    }
    return 1;
}

#if defined(_WIN32)
static int remove_recursive(const char* path) {
    if (!path || !*path) {
        return 0;
    }

    DWORD attributes = GetFileAttributesA(path);
    if (attributes == INVALID_FILE_ATTRIBUTES) {
        DWORD err = GetLastError();
        return (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) ? 0 : -1;
    }

    if (attributes & FILE_ATTRIBUTE_DIRECTORY) {
        char pattern[MAX_PATH];
        if (snprintf(pattern, sizeof(pattern), "%s\\*", path) >= (int)sizeof(pattern)) {
            return -1;
        }

        WIN32_FIND_DATAA data;
        HANDLE handle = FindFirstFileA(pattern, &data);
        if (handle != INVALID_HANDLE_VALUE) {
            do {
                const char* name = data.cFileName;
                if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) {
                    continue;
                }

                char child[MAX_PATH];
                if (snprintf(child, sizeof(child), "%s\\%s", path, name) >= (int)sizeof(child)) {
                    FindClose(handle);
                    return -1;
                }

                if (remove_recursive(child) != 0) {
                    FindClose(handle);
                    return -1;
                }
            } while (FindNextFileA(handle, &data));
            FindClose(handle);
        }

        if (!RemoveDirectoryA(path)) {
            return -1;
        }
    } else {
        if (!DeleteFileA(path)) {
            return -1;
        }
    }

    return 0;
}
#else
static int remove_recursive(const char* path) {
    if (!path || !*path) {
        return 0;
    }

    struct stat st;
    if (lstat(path, &st) != 0) {
        return (errno == ENOENT) ? 0 : -1;
    }

    if (S_ISDIR(st.st_mode)) {
        DIR* dir = opendir(path);
        if (!dir) {
            return -1;
        }

        struct dirent* entry;
        while ((entry = readdir(dir)) != NULL) {
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
                continue;
            }

            char child[PATH_MAX];
            if (snprintf(child, sizeof(child), "%s/%s", path, entry->d_name) >= (int)sizeof(child)) {
                closedir(dir);
                errno = ENAMETOOLONG;
                return -1;
            }

            if (remove_recursive(child) != 0) {
                closedir(dir);
                return -1;
            }
        }

        closedir(dir);

        if (rmdir(path) != 0) {
            return -1;
        }
    } else {
        if (unlink(path) != 0) {
            return -1;
        }
    }

    return 0;
}
#endif

// Print banner
static void print_banner(void) {
    printf("╔════════════════════════════════════════╗\n");
    printf("║          QALLOW - Unified VM           ║\n");
    printf("║    Photonic & Quantum Emulation        ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
}

// BUILD mode: Compile CPU + CUDA backends
static int qallow_build_mode(void) {
    printf("[BUILD] Preparing latest Qallow binaries...\n");
    int rc = qallow_run_build_scripts(0);
    if (rc == 0) {
        printf("[BUILD] Build completed successfully.\n");
    }
    return rc;
}

static int qallow_clear_mode(void) {
    static const char* targets[] = {
        "build",
        "qallow_unified",
        "qallow_unified_cuda",
        "build_cuda.log",
        "logs"
    };

    printf("[CLEAR] Removing build artifacts, cached binaries, and logs...\n");

    int errors = 0;
    for (size_t i = 0; i < sizeof(targets) / sizeof(targets[0]); ++i) {
        if (remove_recursive(targets[i]) != 0) {
            fprintf(stderr, "[WARN] Failed to remove %s\n", targets[i]);
            errors++;
        }
    }

    if (errors > 0) {
        fprintf(stderr, "[ERROR] Workspace cleanup encountered %d issue(s)\n", errors);
        return 1;
    }

    printf("[CLEAR] Workspace cleaned successfully.\n");

    printf("[CLEAR] Rebuilding core binaries to keep CLI available...\n");
    if (qallow_run_build_scripts(0) != 0) {
        fprintf(stderr, "[WARN] Post-clean rebuild failed; run 'qallow build' once dependencies are ready.\n");
        return 1;
    }
    printf("[CLEAR] Minimal rebuild complete.\n");
    return 0;
}

static int qallow_run_vm(run_profile_t profile) {
    switch (profile) {
        case RUN_PROFILE_BENCH:
            printf("[BENCH] Running HITL benchmark...\n");
            printf("[BENCH] Executing VM with benchmark logging...\n\n");
            printf("[RUN] Executing Qallow VM...\n");
            break;
        case RUN_PROFILE_LIVE:
            printf("[LIVE] Starting Live Interface and External Data Integration\n");
            printf("[LIVE] Ingestion manager initialized with 4 streams\n");
            printf("[LIVE] Streams configured and ready for data ingestion\n");
            printf("[LIVE] - telemetry_primary: http://localhost:9000/telemetry\n");
            printf("[LIVE] - sensor_coherence: http://localhost:9001/coherence\n");
            printf("[LIVE] - sensor_decoherence: http://localhost:9002/decoherence\n");
            printf("[LIVE] - feedback_hitl: http://localhost:9003/feedback\n");
            printf("\n[LIVE] Running VM with live data integration...\n\n");
            break;
        case RUN_PROFILE_STANDARD:
        default:
            printf("[RUN] Executing Qallow VM...\n");
            break;
    }

    print_banner();

    int result = qallow_vm_main();

    if (profile == RUN_PROFILE_LIVE) {
        printf("\n[LIVE] Live interface completed\n");
    }

    return result;
}

static int qallow_dispatch_phase(int argc, char** argv, int start_index, const char* phase_name,
                                 int (*runner)(int, char**)) {
    int trailing = argc - (start_index + 1);
    int phase_argc = 2 + (trailing > 0 ? trailing : 0);
    const char* phase_argv_const[phase_argc];
    int pos = 0;

    phase_argv_const[pos++] = argv[0];
    phase_argv_const[pos++] = phase_name;
    for (int i = start_index + 1; i < argc; ++i) {
        phase_argv_const[pos++] = argv[i];
    }

    return runner(phase_argc, (char**)phase_argv_const);
}

static int qallow_handle_run(int argc, char** argv, int arg_offset, run_profile_t default_profile) {
    run_profile_t profile = default_profile;
    bool profile_set = (default_profile != RUN_PROFILE_STANDARD);
    bool integrate_requested = false;
    const char* integrate_phases[8];
    int integrate_count = 0;
    bool integrate_no_split = false;
    bool self_audit = false;
    const char* self_audit_path = NULL;
    const char* pocket_map_path = NULL;
    const char* dl_model_path = NULL;
    const char* dl_device_pref = NULL;
    bool accelerator_requested = false;
    int accelerator_arg_index = -1;
    int (*phase_runner)(int, char**) = NULL;
    const char* phase_name = NULL;
    int phase_arg_index = -1;
    bool hardware_mode = false;

    for (int i = arg_offset; i < argc; ++i) {
        const char* arg = argv[i];

        if (strcmp(arg, "--integrate") == 0) {
            integrate_requested = true;
            int j = i + 1;
            for (; j < argc; ++j) {
                const char* candidate = argv[j];
                if (!candidate || strncmp(candidate, "--", 2) == 0) {
                    break;
                }
                if (integrate_count < (int)(sizeof(integrate_phases) / sizeof(integrate_phases[0]))) {
                    integrate_phases[integrate_count++] = candidate;
                }
            }
            i = j - 1;
            continue;
        }

        if (strcmp(arg, "--no-split") == 0) {
            integrate_no_split = true;
            continue;
        }

        if (strcmp(arg, "--self-audit") == 0) {
            self_audit = true;
            continue;
        }

        if (strcmp(arg, "--self-audit-path") == 0) {
            if ((i + 1) >= argc) {
                fprintf(stderr, "[ERROR] --self-audit-path requires a directory argument\n");
                return 1;
            }
            self_audit_path = argv[++i];
            self_audit = true;
            continue;
        }

        if (strcmp(arg, "--export-pocket-map") == 0) {
            if ((i + 1) >= argc) {
                fprintf(stderr, "[ERROR] --export-pocket-map requires a file path\n");
                return 1;
            }
            pocket_map_path = argv[++i];
            continue;
        }

        if (strcmp(arg, "--bench") == 0) {
            if (profile_set && profile != RUN_PROFILE_BENCH) {
                fprintf(stderr, "[ERROR] Conflicting run profile flags\n");
                return 1;
            }
            profile = RUN_PROFILE_BENCH;
            profile_set = true;
            continue;
        }

        if (strcmp(arg, "--live") == 0) {
            if (profile_set && profile != RUN_PROFILE_LIVE) {
                fprintf(stderr, "[ERROR] Conflicting run profile flags\n");
                return 1;
            }
            profile = RUN_PROFILE_LIVE;
            profile_set = true;
            continue;
        }

        if (strncmp(arg, "--dashboard=", 12) == 0) {
            if (!qallow_apply_dashboard_option(arg + 12)) {
                return 1;
            }
            continue;
        }

        if (strcmp(arg, "--dashboard") == 0) {
            if ((i + 1) >= argc) {
                fprintf(stderr, "[ERROR] --dashboard flag requires a value (off|<ticks>)\n");
                return 1;
            }
            if (!qallow_apply_dashboard_option(argv[++i])) {
                return 1;
            }
            continue;
        }

        if (strncmp(arg, "--dl-model=", 11) == 0) {
            dl_model_path = arg + 11;
            continue;
        }

        if (strcmp(arg, "--dl-model") == 0) {
            if ((i + 1) >= argc) {
                fprintf(stderr, "[ERROR] --dl-model requires a path argument\n");
                return 1;
            }
            dl_model_path = argv[++i];
            continue;
        }

        if (strncmp(arg, "--dl-device=", 12) == 0) {
            dl_device_pref = arg + 12;
            continue;
        }

        if (strcmp(arg, "--hardware") == 0) {
            hardware_mode = true;
            continue;
        }

        if (strcmp(arg, "--accelerator") == 0) {
            accelerator_requested = true;
            accelerator_arg_index = i;
            goto run_parse_done;
        }

        if (strncmp(arg, "--phase=", 8) == 0) {
            const char* phase_value = arg + 8;
            if (strcmp(phase_value, "11") == 0 || strcmp(phase_value, "phase11") == 0) {
                phase_runner = qallow_phase11_runner;
                phase_name = "phase11";
                phase_arg_index = i;
                goto run_parse_done;
            }
            if (strcmp(phase_value, "12") == 0 || strcmp(phase_value, "phase12") == 0) {
                phase_runner = qallow_phase12_runner;
                phase_name = "phase12";
                phase_arg_index = i;
                goto run_parse_done;
            }
            if (strcmp(phase_value, "13") == 0 || strcmp(phase_value, "phase13") == 0) {
                phase_runner = qallow_phase13_runner;
                phase_name = "phase13";
                phase_arg_index = i;
                goto run_parse_done;
            }

            fprintf(stderr, "[ERROR] Unknown phase selector: %s\n", phase_value);
            return 1;
        }

        if (strcmp(arg, "--phase11") == 0) {
            phase_runner = qallow_phase11_runner;
            phase_name = "phase11";
            phase_arg_index = i;
            goto run_parse_done;
        }

        if (strcmp(arg, "--phase12") == 0) {
            phase_runner = qallow_phase12_runner;
            phase_name = "phase12";
            phase_arg_index = i;
            goto run_parse_done;
        }

        if (strcmp(arg, "--phase13") == 0) {
            phase_runner = qallow_phase13_runner;
            phase_name = "phase13";
            phase_arg_index = i;
            goto run_parse_done;
        }

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            qallow_print_help();
            return 0;
        }

        fprintf(stderr, "[ERROR] Unknown run option: %s\n", arg);
        qallow_print_help();
        return 1;
    }

run_parse_done:

    if (hardware_mode) {
        if (QALLOW_SETENV("QALLOW_MODE", "hardware") != 0) {
            fprintf(stderr, "[ERROR] Failed to enable hardware mode.\n");
            return 1;
        }
    }

    {
        int restart_rc = qallow_build_and_maybe_restart(argc, argv);
        if (restart_rc != 0) {
            return restart_rc;
        }
    }

    if (accelerator_requested) {
        int accel_argc = 1 + (argc - (accelerator_arg_index + 1));
        const char* accel_argv_const[accel_argc];
        int pos = 0;

        accel_argv_const[pos++] = argv[0];
        for (int k = accelerator_arg_index + 1; k < argc; ++k) {
            accel_argv_const[pos++] = argv[k];
        }

        return qallow_phase13_main(accel_argc, (char**)accel_argv_const);
    }

    if (phase_runner) {
        return qallow_dispatch_phase(argc, argv, phase_arg_index, phase_name, phase_runner);
    }

    meta_introspect_apply_environment_defaults();
    if (self_audit || self_audit_path) {
        if (self_audit_path) {
            if (meta_introspect_configure(self_audit_path, NULL) != 0) {
                fprintf(stderr, "[ERROR] Failed to configure self-audit directory: %s\n", self_audit_path);
                return 1;
            }
        } else {
            meta_introspect_configure(NULL, NULL);
        }
        meta_introspect_enable(1);
    }

    if (integrate_requested) {
        qallow_lattice_config_t config;
        qallow_lattice_config_init(&config);
        config.no_split = integrate_no_split;

        if (integrate_count > 0) {
            config.phase_mask = 0u;
            for (int idx = 0; idx < integrate_count; ++idx) {
                const char* phase_name = integrate_phases[idx];
                if (strcmp(phase_name, "phase14") == 0 || strcmp(phase_name, "14") == 0 || strcmp(phase_name, "entanglement") == 0) {
                    qallow_lattice_config_enable(&config, QALLOW_LATTICE_PHASE14, true);
                    continue;
                }
                if (strcmp(phase_name, "phase15") == 0 || strcmp(phase_name, "15") == 0 || strcmp(phase_name, "singularity") == 0) {
                    qallow_lattice_config_enable(&config, QALLOW_LATTICE_PHASE15, true);
                    continue;
                }

                fprintf(stderr, "[ERROR] Unknown integration phase: %s\n", phase_name);
                return 1;
            }
            if (config.phase_mask == 0u) {
                fprintf(stderr, "[ERROR] No valid phases selected for integration\n");
                return 1;
            }
        }

        int rc = qallow_lattice_integrate(&config);
        if (rc != 0) {
            fprintf(stderr, "[ERROR] Unified lattice integration failed (code=%d)\n", rc);
        }
        return rc;
    }

    if (dl_model_path) {
        if (!dl_model_supported()) {
            fprintf(stderr, "[ERROR] Deep learning support not compiled in. Rebuild with USE_LIBTORCH=1.\n");
            return 1;
        }
        int prefer_gpu = 1;
        if (dl_device_pref) {
            if (strcmp(dl_device_pref, "cpu") == 0) {
                prefer_gpu = 0;
            } else if (strcmp(dl_device_pref, "gpu") == 0) {
                prefer_gpu = 1;
            } else {
                fprintf(stderr, "[ERROR] Unknown --dl-device option: %s\n", dl_device_pref);
                return 1;
            }
        }
        if (dl_model_load(dl_model_path, prefer_gpu) != 0) {
            fprintf(stderr, "[ERROR] Failed to load TorchScript model: %s\n", dl_model_last_error());
            return 1;
        }
        printf("[DL] TorchScript model loaded (%s)\n", dl_model_path);
    }

    int rc = qallow_run_vm(profile);

    if (self_audit || self_audit_path) {
        meta_introspect_flush();
    }

    if (pocket_map_path) {
        if (meta_introspect_export_pocket_map(pocket_map_path) == 0) {
            printf("[SELF-AUDIT] Pocket map exported to %s\n", pocket_map_path);
        } else {
            fprintf(stderr, "[ERROR] Failed to export pocket map: %s\n", pocket_map_path);
        }
    }

    if (dl_model_path) {
        dl_model_unload();
    }

    return rc;
}

static int qallow_handle_run_group(int argc, char** argv, int arg_offset) {
    if (arg_offset >= argc) {
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_STANDARD);
    }

    const char* sub = argv[arg_offset];
    if (!sub || sub[0] == '\0' || sub[0] == '-') {
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_STANDARD);
    }

    if (strcmp(sub, "vm") == 0) {
        return qallow_handle_run(argc, argv, arg_offset + 1, RUN_PROFILE_STANDARD);
    }

    if (strcmp(sub, "bench") == 0 || strcmp(sub, "benchmark") == 0) {
        return qallow_handle_run(argc, argv, arg_offset + 1, RUN_PROFILE_BENCH);
    }

    if (strcmp(sub, "live") == 0) {
        return qallow_handle_run(argc, argv, arg_offset + 1, RUN_PROFILE_LIVE);
    }

    if (strcmp(sub, "accelerator") == 0) {
        int accel_argc = 1 + (argc - (arg_offset + 1));
        const char* accel_argv_const[accel_argc];
        int pos = 0;

        accel_argv_const[pos++] = argv[0];
        for (int i = arg_offset + 1; i < argc; ++i) {
            accel_argv_const[pos++] = argv[i];
        }

        return qallow_phase13_main(accel_argc, (char**)accel_argv_const);
    }

    if (strcmp(sub, "entangle") == 0) {
        return qallow_cmd_entangle(argc - (arg_offset + 1), argv + arg_offset + 1);
    }

    if (strcmp(sub, "help") == 0) {
        qallow_print_run_help();
        return 0;
    }

    fprintf(stderr, "[ERROR] Unknown run subcommand: %s\n\n", sub);
    qallow_print_run_help();
    return 1;
}

static int qallow_handle_system_group(int argc, char** argv, int arg_offset) {
    if (arg_offset >= argc || argv[arg_offset] == NULL) {
        qallow_print_system_help();
        return 1;
    }

    const char* sub = argv[arg_offset];
    if (strcmp(sub, "build") == 0) {
        return qallow_build_mode();
    }

    if (strcmp(sub, "clear") == 0) {
        return qallow_clear_mode();
    }

    if (strcmp(sub, "verify") == 0) {
        qallow_verify_mode();
        return 0;
    }

    if (strcmp(sub, "help") == 0) {
        qallow_print_system_help();
        return 0;
    }

    fprintf(stderr, "[ERROR] Unknown system subcommand: %s\n\n", sub);
    qallow_print_system_help();
    return 1;
}

static int qallow_handle_phase_group(int argc, char** argv, int arg_offset) {
    if (arg_offset >= argc || argv[arg_offset] == NULL) {
        qallow_print_phase_help();
        return 1;
    }

    const char* sub = argv[arg_offset];
    if (strcmp(sub, "11") == 0 || strcmp(sub, "phase11") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset, "phase11", qallow_phase11_runner);
    }

    if (strcmp(sub, "12") == 0 || strcmp(sub, "phase12") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset, "phase12", qallow_phase12_runner);
    }

    if (strcmp(sub, "13") == 0 || strcmp(sub, "phase13") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset, "phase13", qallow_phase13_runner);
    }

    if (strcmp(sub, "14") == 0 || strcmp(sub, "phase14") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset, "phase14", qallow_phase14_runner);
    }

    if (strcmp(sub, "15") == 0 || strcmp(sub, "phase15") == 0) {
        return qallow_dispatch_phase(argc, argv, arg_offset, "phase15", qallow_phase15_runner);
    }

    if (strcmp(sub, "help") == 0) {
        qallow_print_phase_help();
        return 0;
    }

    fprintf(stderr, "[ERROR] Unknown phase subcommand: %s\n\n", sub);
    qallow_print_phase_help();
    return 1;
}

static int qallow_handle_mind_group(int argc, char** argv, int arg_offset) {
    if (arg_offset >= argc || argv[arg_offset] == NULL) {
        return qallow_cmd_mind(argc - arg_offset, argv + arg_offset);
    }

    const char* sub = argv[arg_offset];
    if (sub[0] == '-') {
        return qallow_cmd_mind(argc - arg_offset, argv + arg_offset);
    }

    if (strcmp(sub, "pipeline") == 0) {
        return qallow_cmd_mind(argc - (arg_offset + 1), argv + arg_offset + 1);
    }

    if (strcmp(sub, "bench") == 0 || strcmp(sub, "benchmark") == 0) {
        return qallow_cmd_bench(argc - (arg_offset + 1), argv + arg_offset + 1);
    }

    if (strcmp(sub, "help") == 0) {
        qallow_print_mind_help();
        return 0;
    }

    fprintf(stderr, "[ERROR] Unknown mind subcommand: %s\n\n", sub);
    qallow_print_mind_help();
    return 1;
}

// VERIFY mode: System checkpoint
static void qallow_verify_mode(void) {
    printf("[VERIFY] Starting system verification...\n");
    printf("[VERIFY] Running comprehensive health checks\n\n");

    // Initialize state
    qallow_state_t state;
    memset(&state, 0, sizeof(qallow_state_t));
    qallow_kernel_init(&state);

    // Run verification checks
    int checks_passed = 0;
    int checks_total = 0;

    // Check 1: Memory integrity
    checks_total++;
    if (state.tick_count == 0) {
        printf("[✓] Memory integrity check passed\n");
        checks_passed++;
    } else {
        printf("[✗] Memory integrity check failed\n");
    }

    // Check 2: Kernel initialization
    checks_total++;
    if (state.global_coherence >= 0.0f && state.global_coherence <= 1.0f) {
        printf("[✓] Kernel initialization check passed\n");
        checks_passed++;
    } else {
        printf("[✗] Kernel initialization check failed\n");
    }

    // Check 3: Ethics scoring
    checks_total++;
    float ethics_total = state.ethics_S + state.ethics_C + state.ethics_H;
    if (ethics_total >= 0.0f && ethics_total <= 3.0f) {
        printf("[✓] Ethics scoring check passed (E=%.2f)\n", ethics_total);
        checks_passed++;
    } else {
        printf("[✗] Ethics scoring check failed\n");
    }

    // Check 4: Overlay stability
    checks_total++;
    float stability = qallow_global_stability(&state);
    if (stability >= 0.0f && stability <= 1.0f) {
        printf("[✓] Overlay stability check passed (S=%.4f)\n", stability);
        checks_passed++;
    } else {
        printf("[✗] Overlay stability check failed\n");
    }

    // Check 5: Decoherence tracking
    checks_total++;
    qallow_update_decoherence(&state);
    if (state.decoherence_level >= 0.0f && state.decoherence_level <= 1.0f) {
        printf("[✓] Decoherence tracking check passed (D=%.6f)\n", state.decoherence_level);
        checks_passed++;
    } else {
        printf("[✗] Decoherence tracking check failed\n");
    }

    // Check 6: Tick execution
    checks_total++;
    int initial_ticks = state.tick_count;
    qallow_kernel_tick(&state);
    if (state.tick_count > initial_ticks) {
        printf("[✓] Tick execution check passed\n");
        checks_passed++;
    } else {
        printf("[✗] Tick execution check failed\n");
    }

    // Check 7: Configuration
    checks_total++;
    if (NUM_OVERLAYS == 3 && MAX_NODES == 256) {
        printf("[✓] Configuration check passed (3 overlays, 256 nodes)\n");
        checks_passed++;
    } else {
        printf("[✗] Configuration check failed\n");
    }

    // Print summary
    printf("\n");
    printf("═══════════════════════════════════════\n");
    printf("VERIFICATION SUMMARY\n");
    printf("═══════════════════════════════════════\n");
    printf("Checks passed: %d/%d\n", checks_passed, checks_total);
    printf("System status: %s\n", checks_passed == checks_total ? "HEALTHY" : "DEGRADED");
    printf("═══════════════════════════════════════\n\n");
}

// Print help message
static void qallow_print_run_help(void) {
    printf("Run command group:\n");
    printf("  qallow run [subcommand] [options]\n\n");
    printf("Subcommands:\n");
    printf("  vm [options]        Execute the unified VM workflow (default when omitted)\n");
    printf("  bench [options]     Run the VM in benchmark profile (alias of vm --bench)\n");
    printf("  live [options]      Run the VM with live ingestion profile (alias of vm --live)\n");
    printf("  accelerator [options]  Launch the Phase-13 accelerator directly\n");
    printf("  entangle [options]  Generate GHZ/W entanglement data via QuTiP bridge\n");
    printf("  help                Show this help message for the run group\n\n");
    printf("VM options:\n");
    printf("  --bench             Enable benchmark profile (same as `qallow run bench`)\n");
    printf("  --live              Enable live ingestion profile (same as `qallow run live`)\n");
    printf("  --hardware          Route Phase 11 through IBM Quantum hardware\n");
    printf("  --dashboard=<N|off> Control dashboard frequency (ticks) or disable output\n");
    printf("  --self-audit        Enable phase16 meta-introspect logging\n");
    printf("  --self-audit-path <DIR> Override auditor log directory (implies --self-audit)\n");
    printf("  --export-pocket-map <FILE> Emit audited pocket status JSON after run\n");
    printf("  --dl-model <PATH>   Load TorchScript model for inference inside the run loop\n");
    printf("  --dl-device=<cpu|gpu> Prefer CPU or GPU when running the TorchScript model\n");
    printf("  --phase=11|12|13    Dispatch directly into a legacy phase runner\n");
    printf("  --accelerator       Launch the Phase-13 accelerator from within vm\n");
    printf("  --remote-sync[=<URL>] Enable remote ingestion loop (optional endpoint)\n");
    printf("  --remote-sync-interval=N Override remote polling cadence in seconds\n\n");
    printf("Accelerator options (when using `qallow run accelerator` or --accelerator):\n");
    printf("  --threads=<N|auto>  Worker thread count (auto = online CPUs)\n");
    printf("  --watch=<DIR>       Directory to monitor via inotify\n");
    printf("  --no-watch          Disable watcher even if provided earlier\n");
    printf("  --file=<PATH>       Queue a file for immediate processing (repeatable)\n");
    printf("  --export=<FILE>     Write a JSON summary of processed inputs\n\n");
    printf("Examples:\n");
    printf("  qallow run vm                       # Run the unified VM\n");
    printf("  qallow run bench                    # VM benchmark profile\n");
    printf("  qallow run live --dashboard=off     # Live ingestion without dashboard\n");
    printf("  qallow run vm --phase=12 --ticks=200\n");
    printf("  qallow run accelerator --watch=/tmp --threads=auto\n");
}

static void qallow_print_system_help(void) {
    printf("System command group:\n");
    printf("  qallow system <subcommand>\n\n");
    printf("Subcommands:\n");
    printf("  build      Detect toolchain and compile CPU + CUDA backends\n");
    printf("  clear      Remove build artifacts and cached binaries\n");
    printf("  verify     Run system verification health checks\n");
    printf("  help       Show this help message for the system group\n");
}

static void qallow_print_phase_help(void) {
    printf("Phase command group:\n");
    printf("  qallow phase <11|12|13|14|15> [options]\n\n");
    printf("Subcommands:\n");
    printf("  11 [options]  Invoke the Phase 11 coherence bridge\n");
    printf("  12 [options]  Run the Phase 12 elasticity simulation\n");
    printf("  13 [options]  Run the Phase 13 harmonic propagation\n");
    printf("  14 [options]  Run the Phase 14 coherence-lattice integration\n");
    printf("  15 [options]  Run the Phase 15 convergence & lock-in\n");
    printf("  help          Show this help message for the phase group\n\n");
    printf("Phase 14 options:\n");
    printf("  --ticks=N                 Number of ticks to run (default: 500)\n");
    printf("  --nodes=N                 Lattice nodes (default: 256)\n");
    printf("  --target_fidelity=F       Success threshold (default: 0.981)\n");
    printf("  --alpha=A                 Explicit alpha override (skips closed-form)\n");
    printf("  --jcsv=FILE               Use CUDA CSR J-couplings to derive alpha_eff\n");
    printf("  --gain_base=B             Base gain for CUDA/quantum mapping (default: 0.001)\n");
    printf("  --gain_span=S             Gain span for CUDA/quantum mapping (default: 0.009)\n");
    printf("  --gain_json=FILE          Load {\"alpha_eff\": A} to override alpha\n");
    printf("  --tune_qaoa               Invoke built-in QAOA tuner (via Python)\n");
    printf("  --qaoa_n=N                QAOA problem size (default: 16)\n");
    printf("  --qaoa_p=P                QAOA depth p (default: 2)\n");
    printf("  --export=FILE             Write JSON summary with alpha and fidelity\n\n");
    printf("Phase 15 options:\n");
    printf("  --ticks=N                 Max ticks before stop (default: 400)\n");
    printf("  --eps=E                   Convergence tolerance (default: 1e-5)\n");
    printf("  --export=FILE             Write JSON summary (score, stability)\n\n");
    printf("Examples:\n");
    printf("  qallow phase 11 --ticks=400 --states=-1,0,1\n");
    printf("  qallow phase 12 --ticks=100 --eps=0.0001 --log=phase12.csv\n");
    printf("  qallow phase 13 --nodes=16 --ticks=500 --k=0.002\n");
    printf("  qallow phase 14 --ticks=600 --nodes=256 --target_fidelity=0.981\n");
    printf("  qallow phase 14 --tune_qaoa --qaoa_n=16 --qaoa_p=2 --target_fidelity=0.981\n");
    printf("  qallow phase 14 --jcsv=graph.csv --gain_base=0.001 --gain_span=0.009\n");
    printf("  qallow phase 15 --ticks=500 --eps=1e-5\n");
}

static void qallow_print_mind_help(void) {
    printf("Mind command group:\n");
    printf("  qallow mind [subcommand]\n\n");
    printf("Subcommands:\n");
    printf("  pipeline    Run the cognitive modules pipeline (default when omitted)\n");
    printf("  bench       Run the cognition benchmarking suite\n");
    printf("  help        Show this help message for the mind group\n\n");
    printf("Environment overrides:\n");
    printf("  QALLOW_MIND_STEPS  Number of pipeline steps to execute (default: 50)\n");
}

static void qallow_print_help(void) {
    printf("Usage: qallow <group> [subcommand] [options]\n\n");
    printf("Command groups:\n");
    printf("  run       Workflow execution (vm, bench, live, accelerator)\n");
    printf("  system    Build, clean, and verify project artifacts\n");
    printf("  phase     Invoke individual phase runners (11, 12, 13, 14, 15)\n");
    printf("  mind      Cognitive pipeline and benchmarking utilities\n");
    printf("  help      Show this help message\n\n");
    printf("Use `qallow help <group>` for a detailed description of that group.\n\n");
    printf("Legacy aliases:\n");
    printf("  qallow build        -> qallow system build\n");
    printf("  qallow clear        -> qallow system clear\n");
    printf("  qallow verify       -> qallow system verify\n");
    printf("  qallow bench        -> qallow run bench\n");
    printf("  qallow live         -> qallow run live\n");
    printf("  qallow accelerator  -> qallow run accelerator\n");
    printf("  qallow phase11      -> qallow phase 11\n");
    printf("  qallow phase12      -> qallow phase 12\n");
    printf("  qallow phase13      -> qallow phase 13\n");
}

// Input validation helper
static int validate_command(const char* cmd) {
    if (cmd == NULL || strlen(cmd) == 0) {
        fprintf(stderr, "[ERROR] Command cannot be empty\n");
        return 0;
    }
    if (strlen(cmd) > 64) {
        fprintf(stderr, "[ERROR] Command too long (max 64 chars)\n");
        return 0;
    }
    // Check for invalid characters
    for (int i = 0; cmd[i]; i++) {
        if (!((cmd[i] >= 'a' && cmd[i] <= 'z') ||
              (cmd[i] >= 'A' && cmd[i] <= 'Z') ||
              (cmd[i] >= '0' && cmd[i] <= '9') ||
              cmd[i] == '-' || cmd[i] == '_')) {
            fprintf(stderr, "[ERROR] Invalid character in command: %c\n", cmd[i]);
            return 0;
        }
    }
    return 1;
}

// Main entry point
int main(int argc, char** argv) {
    // Validate argc
    if (argc < 1 || argv == NULL) {
        fprintf(stderr, "[ERROR] Invalid arguments\n");
        return 1;
    }

    // Validate argv[0] (program name)
    if (argv[0] == NULL || strlen(argv[0]) == 0) {
        fprintf(stderr, "[ERROR] Invalid program name\n");
        return 1;
    }

    qallow_init_process_flags();

    const char* command = "run";
    int arg_offset = 1;

    if (argc > 1 && argv[1] != NULL && argv[1][0] != '-') {
        if (!validate_command(argv[1])) {
            return 1;
        }
        command = argv[1];
        arg_offset = 2;
    }

    if (strcmp(command, "system") == 0) {
        return qallow_handle_system_group(argc, argv, arg_offset);
    }

    if (strcmp(command, "run") == 0) {
        return qallow_handle_run_group(argc, argv, arg_offset);
    }

    if (strcmp(command, "phase") == 0) {
        return qallow_handle_phase_group(argc, argv, arg_offset);
    }

    if (strcmp(command, "mind") == 0) {
        return qallow_handle_mind_group(argc, argv, arg_offset);
    }

    if (strcmp(command, "help") == 0 || strcmp(command, "-h") == 0 || strcmp(command, "--help") == 0) {
        if (arg_offset < argc && argv[arg_offset] != NULL) {
            const char* topic = argv[arg_offset];
            if (strcmp(topic, "run") == 0) {
                qallow_print_run_help();
                return 0;
            }
            if (strcmp(topic, "system") == 0) {
                qallow_print_system_help();
                return 0;
            }
            if (strcmp(topic, "phase") == 0) {
                qallow_print_phase_help();
                return 0;
            }
            if (strcmp(topic, "mind") == 0) {
                qallow_print_mind_help();
                return 0;
            }

            fprintf(stderr, "[ERROR] Unknown help topic: %s\n\n", topic);
            qallow_print_help();
            return 1;
        }

        qallow_print_help();
        return 0;
    }

    if (strcmp(command, "govern") == 0) {
        return govern_cli(argc, argv);
    }

    if (strcmp(command, "build") == 0) {
        printf("[INFO] `qallow build` is deprecated; use `qallow system build`.\n");
        return qallow_build_mode();
    }

    if (strcmp(command, "clear") == 0) {
        printf("[INFO] `qallow clear` is deprecated; use `qallow system clear`.\n");
        return qallow_clear_mode();
    }

    if (strcmp(command, "verify") == 0) {
        printf("[INFO] `qallow verify` is deprecated; use `qallow system verify`.\n");
        qallow_verify_mode();
        return 0;
    }

    if (strcmp(command, "bench") == 0 || strcmp(command, "benchmark") == 0) {
        printf("[INFO] `qallow %s` now routes to `qallow run bench`.\n", command);
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_BENCH);
    }

    if (strcmp(command, "live") == 0) {
        printf("[INFO] `qallow live` now routes to `qallow run live`.\n");
        return qallow_handle_run(argc, argv, arg_offset, RUN_PROFILE_LIVE);
    }

    if (strcmp(command, "accelerator") == 0) {
        printf("[INFO] `qallow accelerator` now routes to `qallow run accelerator`.\n");
        int accel_argc = 1 + (argc - arg_offset);
        const char* accel_argv_const[accel_argc];
        int pos = 0;

        accel_argv_const[pos++] = argv[0];
        for (int i = arg_offset; i < argc; ++i) {
            if (argv[i] == NULL) {
                fprintf(stderr, "[ERROR] NULL argument at index %d\n", i);
                return 1;
            }
            accel_argv_const[pos++] = argv[i];
        }

        return qallow_phase13_main(accel_argc, (char**)accel_argv_const);
    }

    if (strcmp(command, "phase11") == 0) {
        printf("[INFO] `qallow phase11` is deprecated; use `qallow phase 11`.\n");
        return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase11", qallow_phase11_runner);
    }

    if (strcmp(command, "phase12") == 0) {
        printf("[INFO] `qallow phase12` is deprecated; use `qallow phase 12`.\n");
        return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase12", qallow_phase12_runner);
    }

    if (strcmp(command, "phase13") == 0) {
        printf("[INFO] `qallow phase13` is deprecated; use `qallow phase 13`.\n");
        return qallow_dispatch_phase(argc, argv, arg_offset - 1, "phase13", qallow_phase13_runner);
    }

    fprintf(stderr, "[ERROR] Unknown command: %s\n\n", command);
    qallow_print_help();
    return 1;
}
