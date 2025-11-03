#include "qallow_entanglement.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>

#ifndef QALLOW_PYTHON_MODULE
#define QALLOW_PYTHON_MODULE "python.quantum.ghz_w_sim"
#endif

static qallow_entanglement_snapshot_t g_cached_snapshot;
static int g_cached_valid = 0;

static const char* detect_python_binary(void) {
    const char* env_python = getenv("QALLOW_PYTHON");
    if (env_python && *env_python && access(env_python, X_OK) == 0) {
        return env_python;
    }
    if (access("python3", X_OK) == 0) {
        return "python3";
    }
    if (access("/usr/bin/python3", X_OK) == 0) {
        return "/usr/bin/python3";
    }
    return "python";
}

static void trim(char* s) {
    if (!s) {
        return;
    }
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' || s[len - 1] == ' ' || s[len - 1] == '\t')) {
        s[len - 1] = '\0';
        len--;
    }
    size_t start = 0U;
    while (s[start] == ' ' || s[start] == '\t') {
        start++;
    }
    if (start > 0) {
        memmove(s, s + start, strlen(s + start) + 1);
    }
}

static int parse_probabilities(const char* value, double* out, int max_count) {
    if (!value || !out || max_count <= 0) {
        return 0;
    }
    int count = 0;
    const char* cursor = value;
    while (*cursor && count < max_count) {
        char* end_ptr = NULL;
        double v = strtod(cursor, &end_ptr);
        if (cursor == end_ptr) {
            break;
        }
        out[count++] = v;
        if (!end_ptr || *end_ptr == '\0') {
            break;
        }
        if (*end_ptr == ',') {
            cursor = end_ptr + 1;
        } else {
            cursor = end_ptr;
        }
    }
    return count;
}

qallow_entanglement_state_t qallow_entanglement_state_from_string(const char* raw) {
    if (!raw || !*raw) {
        return QALLOW_ENT_STATE_GHZ;
    }
    char buf[16];
    size_t n = strlen(raw);
    if (n >= sizeof(buf)) {
        n = sizeof(buf) - 1;
    }
    for (size_t i = 0; i < n; ++i) {
        buf[i] = (char)tolower((unsigned char)raw[i]);
    }
    buf[n] = '\0';
    if (strcmp(buf, "w") == 0 || strcmp(buf, "wstate") == 0) {
        return QALLOW_ENT_STATE_W;
    }
    return QALLOW_ENT_STATE_GHZ;
}

const char* qallow_entanglement_state_name(qallow_entanglement_state_t state) {
    switch (state) {
        case QALLOW_ENT_STATE_W:
            return "w";
        case QALLOW_ENT_STATE_GHZ:
        default:
            return "ghz";
    }
}

const qallow_entanglement_snapshot_t* qallow_entanglement_get_cached(void) {
    return g_cached_valid ? &g_cached_snapshot : NULL;
}

int qallow_entanglement_generate(
    qallow_entanglement_snapshot_t* out,
    qallow_entanglement_state_t state,
    int qubits,
    int validate
) {
    if (!out) {
        return -1;
    }
    if (qubits <= 1 || qubits > 5) {
        qubits = 4;
    }

    const char* python = detect_python_binary();
    const char* state_name = qallow_entanglement_state_name(state);

    char command[512];
    int written = snprintf(
        command,
        sizeof(command),
        "\"%s\" -m %s --state=%s --qubits=%d %s",
        python,
        QALLOW_PYTHON_MODULE,
        state_name,
        qubits,
        validate ? "--validate" : ""
    );
    if (written < 0 || (size_t)written >= sizeof(command)) {
        return -1;
    }

    FILE* pipe = popen(command, "r");
    if (!pipe) {
        return -1;
    }

    qallow_entanglement_snapshot_t snapshot = {0};
    snapshot.state = state;
    snapshot.qubits = qubits;
    snapshot.amplitude_count = 0;
    snapshot.fidelity = 0.0;
    snprintf(snapshot.backend, sizeof(snapshot.backend), "%s", "unknown");

    char line[512];
    while (fgets(line, sizeof(line), pipe)) {
        trim(line);
        if (line[0] == '\0' || line[0] == '#') {
            continue;
        }
        char* eq = strchr(line, '=');
        if (!eq) {
            continue;
        }
        *eq = '\0';
        char* key = line;
        char* value = eq + 1;
        trim(key);
        trim(value);

        if (strcmp(key, "STATE") == 0) {
            snapshot.state = qallow_entanglement_state_from_string(value);
        } else if (strcmp(key, "QUBITS") == 0) {
            snapshot.qubits = atoi(value);
        } else if (strcmp(key, "BACKEND") == 0) {
            snprintf(snapshot.backend, sizeof(snapshot.backend), "%s", value);
        } else if (strcmp(key, "FIDELITY") == 0) {
            snapshot.fidelity = strtod(value, NULL);
        } else if (strcmp(key, "PROBABILITIES") == 0) {
            snapshot.amplitude_count = parse_probabilities(value, snapshot.amplitudes, (int)(sizeof(snapshot.amplitudes) / sizeof(snapshot.amplitudes[0])));
        }
    }

    int status = pclose(pipe);
    if (status == -1) {
        return -1;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        return -1;
    }

    if (snapshot.amplitude_count <= 0) {
        return -1;
    }

    *out = snapshot;
    g_cached_snapshot = snapshot;
    g_cached_valid = 1;
    return 0;
}
