#define _POSIX_C_SOURCE 200809L

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <limits.h>

#define WINDOW_WIDTH 1100
#define WINDOW_HEIGHT 720

#define MAX_METRICS 16
#define TOKEN_LEN 64

#define MAX_LOG_LINES 200
#define MAX_LOG_LEN 256

#define MAX_CMD_ARGS 8
#define MAX_BUTTONS 8

typedef struct {
    char names[MAX_METRICS][TOKEN_LEN];
    double values[MAX_METRICS];
    size_t metric_count;
    int tick;
    char mode[TOKEN_LEN];
} telemetry_snapshot_t;

typedef struct {
    int tick;
    double average_score;
    double average_coherence;
    double average_decoherence;
    double memory_usage_mb;
    double memory_peak_mb;
} pocket_metrics_t;

typedef struct {
    char telemetry_path[PATH_MAX];
    char pocket_metrics_path[PATH_MAX];
    char runner_path[PATH_MAX];
    char repo_root[PATH_MAX];
    const char *font_path;
    Uint32 refresh_interval_ms;
    Uint32 pocket_refresh_ms;
} ui_config_t;

typedef struct {
    telemetry_snapshot_t telemetry;
    pocket_metrics_t pocket;
    SDL_mutex *mutex;
    char status_text[MAX_LOG_LEN];
    char log_lines[MAX_LOG_LINES][MAX_LOG_LEN];
    int log_start;
    int log_size;
} ui_state_t;

typedef struct {
    char label[64];
    char working_dir[PATH_MAX];
    int argc;
    char argv[MAX_CMD_ARGS][PATH_MAX];
} command_request_t;

typedef enum {
    ACTION_NONE = 0,
    ACTION_BUILD_CUDA,
    ACTION_RUN_BINARY,
    ACTION_RUN_ACCELERATOR,
    ACTION_PHASE_14,
    ACTION_PHASE_15,
    ACTION_PHASE_16,
    ACTION_STOP_COMMAND
} action_t;

typedef struct {
    SDL_mutex *mutex;
    bool active;
    bool cancel_requested;
    pid_t pid;
    int exit_code;
    action_t active_action;
    command_request_t request;
    ui_state_t *ui_state;
} command_runner_t;

typedef struct {
    SDL_Rect rect;
    action_t action;
    const char *label;
    const char *hint;
} button_t;

static void trim_newline(char *str) {
    if (!str) {
        return;
    }
    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r')) {
        str[len - 1] = '\0';
        --len;
    }
}

static size_t tokenize_csv_line(const char *line, char tokens[][TOKEN_LEN], size_t max_tokens) {
    size_t count = 0;
    const char *cursor = line;

    while (cursor && *cursor && count < max_tokens) {
        const char *comma = strchr(cursor, ',');
        size_t segment_len = comma ? (size_t)(comma - cursor) : strlen(cursor);

        if (segment_len >= TOKEN_LEN) {
            segment_len = TOKEN_LEN - 1;
        }

        memcpy(tokens[count], cursor, segment_len);
        tokens[count][segment_len] = '\0';
        trim_newline(tokens[count]);

        ++count;
        if (!comma) {
            break;
        }
        cursor = comma + 1;
    }

    return count;
}

static bool read_latest_snapshot(const char *path, telemetry_snapshot_t *snapshot) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        return false;
    }

    char header_line[512];
    if (!fgets(header_line, sizeof(header_line), fp)) {
        fclose(fp);
        return false;
    }
    trim_newline(header_line);

    char last_line[512] = {0};
    char line[512];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\0' || line[0] == '\n') {
            continue;
        }
        strncpy(last_line, line, sizeof(last_line) - 1);
    }
    fclose(fp);

    if (last_line[0] == '\0') {
        return false;
    }

    char header_tokens[MAX_METRICS + 2][TOKEN_LEN];
    char value_tokens[MAX_METRICS + 2][TOKEN_LEN];

    size_t header_count = tokenize_csv_line(header_line, header_tokens, MAX_METRICS + 2);
    size_t value_count = tokenize_csv_line(last_line, value_tokens, MAX_METRICS + 2);
    if (header_count == 0 || header_count != value_count) {
        return false;
    }

    telemetry_snapshot_t updated = {0};

    for (size_t i = 0; i < header_count; ++i) {
        const char *name = header_tokens[i];
        const char *value = value_tokens[i];

        if (i == 0) {
            updated.tick = atoi(value);
        } else if (i == header_count - 1) {
            strncpy(updated.mode, value, sizeof(updated.mode) - 1);
        } else if (updated.metric_count < MAX_METRICS) {
            strncpy(updated.names[updated.metric_count], name, TOKEN_LEN - 1);
            updated.values[updated.metric_count] = strtod(value, NULL);
            ++updated.metric_count;
        }
    }

    *snapshot = updated;
    return true;
}

static double extract_json_double(const char *json, const char *key, double fallback) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) {
        return fallback;
    }
    pos = strchr(pos, ':');
    if (!pos) {
        return fallback;
    }
    return strtod(pos + 1, NULL);
}

static int extract_json_int(const char *json, const char *key, int fallback) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (!pos) {
        return fallback;
    }
    pos = strchr(pos, ':');
    if (!pos) {
        return fallback;
    }
    return atoi(pos + 1);
}

static bool read_pocket_metrics(const char *path, pocket_metrics_t *metrics) {
    FILE *fp = fopen(path, "r");
    if (!fp) {
        return false;
    }

    char buffer[1024];
    size_t read_bytes = fread(buffer, 1, sizeof(buffer) - 1, fp);
    fclose(fp);
    if (read_bytes == 0) {
        return false;
    }
    buffer[read_bytes] = '\0';

    pocket_metrics_t updated = {0};
    updated.tick = extract_json_int(buffer, "tick", 0);
    updated.average_score = extract_json_double(buffer, "average_score", 0.0);
    updated.average_coherence = extract_json_double(buffer, "average_coherence", 0.0);
    updated.average_decoherence = extract_json_double(buffer, "average_decoherence", 0.0);
    updated.memory_usage_mb = extract_json_double(buffer, "memory_usage_mb", 0.0);
    updated.memory_peak_mb = extract_json_double(buffer, "memory_peak_mb", 0.0);

    *metrics = updated;
    return true;
}

static void ui_state_init(ui_state_t *state) {
    memset(state, 0, sizeof(*state));
    state->mutex = SDL_CreateMutex();
    if (state->mutex) {
        strncpy(state->status_text, "Ready", sizeof(state->status_text) - 1);
    }
}

static void ui_state_destroy(ui_state_t *state) {
    if (state->mutex) {
        SDL_DestroyMutex(state->mutex);
        state->mutex = NULL;
    }
}

static void append_log_line(ui_state_t *state, const char *line) {
    if (!state || !state->mutex || !line) {
        return;
    }
    SDL_LockMutex(state->mutex);
    int index;
    if (state->log_size < MAX_LOG_LINES) {
        index = (state->log_start + state->log_size) % MAX_LOG_LINES;
        state->log_size += 1;
    } else {
        index = state->log_start;
        state->log_start = (state->log_start + 1) % MAX_LOG_LINES;
    }
    strncpy(state->log_lines[index], line, MAX_LOG_LEN - 1);
    state->log_lines[index][MAX_LOG_LEN - 1] = '\0';
    SDL_UnlockMutex(state->mutex);
}

static void set_status_text(ui_state_t *state, const char *text) {
    if (!state || !state->mutex || !text) {
        return;
    }
    SDL_LockMutex(state->mutex);
    strncpy(state->status_text, text, sizeof(state->status_text) - 1);
    state->status_text[sizeof(state->status_text) - 1] = '\0';
    SDL_UnlockMutex(state->mutex);
}

static void detect_runner_binary(char *out, size_t out_len) {
    const char *candidates[] = {
        "./build/qallow_unified_cuda",
        "./build/qallow_unified",
        "./qallow_unified_cuda",
        "./qallow_unified"
    };

    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (access(candidates[i], X_OK) == 0) {
            strncpy(out, candidates[i], out_len - 1);
            out[out_len - 1] = '\0';
            return;
        }
    }
    out[0] = '\0';
}

static void parse_arguments(int argc, char **argv, ui_config_t *cfg) {
    memset(cfg, 0, sizeof(*cfg));
    strncpy(cfg->telemetry_path, "data/logs/telemetry_stream.csv", sizeof(cfg->telemetry_path) - 1);
    strncpy(cfg->pocket_metrics_path, "data/telemetry/pocket_metrics.json", sizeof(cfg->pocket_metrics_path) - 1);
    cfg->font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf";
    cfg->refresh_interval_ms = 750;
    cfg->pocket_refresh_ms = 1200;

    if (!getcwd(cfg->repo_root, sizeof(cfg->repo_root))) {
        strncpy(cfg->repo_root, ".", sizeof(cfg->repo_root) - 1);
    }

    detect_runner_binary(cfg->runner_path, sizeof(cfg->runner_path));

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strncmp(arg, "--telemetry=", 12) == 0) {
            strncpy(cfg->telemetry_path, arg + 12, sizeof(cfg->telemetry_path) - 1);
        } else if (strncmp(arg, "--pocket-json=", 14) == 0) {
            strncpy(cfg->pocket_metrics_path, arg + 14, sizeof(cfg->pocket_metrics_path) - 1);
        } else if (strncmp(arg, "--runner=", 9) == 0) {
            strncpy(cfg->runner_path, arg + 9, sizeof(cfg->runner_path) - 1);
        } else if (strncmp(arg, "--font=", 7) == 0) {
            cfg->font_path = arg + 7;
        } else if (strncmp(arg, "--refresh-ms=", 13) == 0) {
            int value = atoi(arg + 13);
            if (value >= 100) {
                cfg->refresh_interval_ms = (Uint32)value;
            }
        } else if (strncmp(arg, "--pocket-refresh-ms=", 20) == 0) {
            int value = atoi(arg + 20);
            if (value >= 200) {
                cfg->pocket_refresh_ms = (Uint32)value;
            }
        } else if (strncmp(arg, "--repo-root=", 12) == 0) {
            strncpy(cfg->repo_root, arg + 12, sizeof(cfg->repo_root) - 1);
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            printf("Qallow Telemetry UI\n\n");
            printf("Options:\n");
            printf("  --telemetry=<path>         Path to telemetry CSV\n");
            printf("  --pocket-json=<path>       Path to pocket metrics JSON\n");
            printf("  --runner=<path>            Path to qallow executable for phase control\n");
            printf("  --font=<path>              Path to TTF font for rendering\n");
            printf("  --repo-root=<path>         Working directory for build/run commands\n");
            printf("  --refresh-ms=<n>           Telemetry refresh interval (>=100)\n");
            printf("  --pocket-refresh-ms=<n>    Pocket metrics refresh interval (>=200)\n");
            printf("  --help, -h                 Show this message\n");
            exit(0);
        }
    }
}

static void command_runner_init(command_runner_t *runner, ui_state_t *state) {
    memset(runner, 0, sizeof(*runner));
    runner->mutex = SDL_CreateMutex();
    runner->pid = -1;
    runner->ui_state = state;
}

static void command_runner_destroy(command_runner_t *runner) {
    if (runner->mutex) {
        SDL_DestroyMutex(runner->mutex);
        runner->mutex = NULL;
    }
}

static void append_logf(ui_state_t *state, const char *fmt, ...) {
    char buffer[MAX_LOG_LEN];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);
    va_end(args);
    append_log_line(state, buffer);
}

static int command_thread(void *userdata) {
    command_runner_t *runner = (command_runner_t *)userdata;
    command_request_t request;

    SDL_LockMutex(runner->mutex);
    request = runner->request;
    runner->pid = -1;
    SDL_UnlockMutex(runner->mutex);

    if (runner->ui_state) {
        append_logf(runner->ui_state, "[CMD] %s", request.label);
        set_status_text(runner->ui_state, request.label);
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        if (runner->ui_state) {
            append_logf(runner->ui_state, "[ERROR] pipe() failed: %s", strerror(errno));
            set_status_text(runner->ui_state, "Command start failed (pipe)");
        }
        SDL_LockMutex(runner->mutex);
        runner->active = false;
        runner->active_action = ACTION_NONE;
        SDL_UnlockMutex(runner->mutex);
        return -1;
    }

    pid_t pid = fork();
    if (pid == 0) {
        // Child
        if (request.working_dir[0] != '\0') {
            if (chdir(request.working_dir) != 0) {
                fprintf(stderr, "qallow_ui: chdir failed: %s\n", strerror(errno));
                _exit(1);
            }
        }

        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);

        char *argv[MAX_CMD_ARGS + 1] = {0};
        for (int i = 0; i < request.argc && i < MAX_CMD_ARGS; ++i) {
            argv[i] = request.argv[i];
        }
        argv[request.argc] = NULL;

        execvp(argv[0], argv);
        fprintf(stderr, "qallow_ui: execvp failed: %s\n", strerror(errno));
        _exit(127);
    } else if (pid < 0) {
        close(pipefd[0]);
        close(pipefd[1]);
        if (runner->ui_state) {
            append_logf(runner->ui_state, "[ERROR] fork() failed: %s", strerror(errno));
            set_status_text(runner->ui_state, "Command start failed (fork)");
        }
        SDL_LockMutex(runner->mutex);
        runner->active = false;
        runner->active_action = ACTION_NONE;
        SDL_UnlockMutex(runner->mutex);
        return -1;
    }

    close(pipefd[1]);

    SDL_LockMutex(runner->mutex);
    runner->pid = pid;
    SDL_UnlockMutex(runner->mutex);

    char buffer[256];
    char partial[MAX_LOG_LEN] = {0};
    size_t partial_len = 0;
    ssize_t bytes;

    while ((bytes = read(pipefd[0], buffer, sizeof(buffer))) > 0) {
        for (ssize_t i = 0; i < bytes; ++i) {
            char c = buffer[i];
            if (c == '\r') {
                continue;
            }
            if (c == '\n') {
                partial[partial_len] = '\0';
                if (partial_len > 0 && runner->ui_state) {
                    append_log_line(runner->ui_state, partial);
                }
                partial_len = 0;
            } else if (partial_len + 1 < sizeof(partial)) {
                partial[partial_len++] = c;
            }
        }
    }
    close(pipefd[0]);

    if (partial_len > 0 && runner->ui_state) {
        partial[partial_len] = '\0';
        append_log_line(runner->ui_state, partial);
    }

    int status = 0;
    pid_t waited = waitpid(pid, &status, 0);
    bool terminated_normally = (waited == pid);
    bool cancelled = false;

    SDL_LockMutex(runner->mutex);
    cancelled = runner->cancel_requested;
    runner->pid = -1;
    runner->exit_code = terminated_normally && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    runner->active = false;
    runner->active_action = ACTION_NONE;
    SDL_UnlockMutex(runner->mutex);

    if (runner->ui_state) {
        if (!terminated_normally) {
            append_logf(runner->ui_state, "[ERROR] Command failed: waitpid error");
            set_status_text(runner->ui_state, "Command failed (waitpid)");
        } else if (cancelled) {
            append_logf(runner->ui_state, "[CMD] Command cancelled");
            set_status_text(runner->ui_state, "Command cancelled");
        } else if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
            append_logf(runner->ui_state, "[CMD] Completed successfully");
            set_status_text(runner->ui_state, "Command completed");
        } else if (WIFEXITED(status)) {
            append_logf(runner->ui_state, "[CMD] Exit code %d", WEXITSTATUS(status));
            set_status_text(runner->ui_state, "Command finished with errors");
        } else if (WIFSIGNALED(status)) {
            append_logf(runner->ui_state, "[CMD] Terminated by signal %d", WTERMSIG(status));
            set_status_text(runner->ui_state, "Command terminated");
        } else {
            append_log_line(runner->ui_state, "[CMD] Command ended unexpectedly");
            set_status_text(runner->ui_state, "Command ended unexpectedly");
        }
    }

    return 0;
}

static bool command_runner_start(command_runner_t *runner, const command_request_t *request, action_t action) {
    if (!runner || !runner->mutex || !request) {
        return false;
    }
    SDL_LockMutex(runner->mutex);
    if (runner->active) {
        SDL_UnlockMutex(runner->mutex);
        if (runner->ui_state) {
            append_log_line(runner->ui_state, "[WARN] Command already running");
            set_status_text(runner->ui_state, "Command already running");
        }
        return false;
    }
    runner->request = *request;
    runner->cancel_requested = false;
    runner->exit_code = -1;
    runner->active = true;
    runner->active_action = action;
    SDL_UnlockMutex(runner->mutex);

    SDL_Thread *thread = SDL_CreateThread(command_thread, "qallow_cmd", runner);
    if (!thread) {
        if (runner->ui_state) {
            append_logf(runner->ui_state, "[ERROR] SDL_CreateThread failed: %s", SDL_GetError());
            set_status_text(runner->ui_state, "Thread creation failed");
        }
        SDL_LockMutex(runner->mutex);
        runner->active = false;
        runner->active_action = ACTION_NONE;
        SDL_UnlockMutex(runner->mutex);
        return false;
    }
    SDL_DetachThread(thread);
    return true;
}

static void command_runner_stop(command_runner_t *runner) {
    if (!runner || !runner->mutex) {
        return;
    }
    pid_t pid_to_kill = -1;

    SDL_LockMutex(runner->mutex);
    if (!runner->active) {
        SDL_UnlockMutex(runner->mutex);
        if (runner->ui_state) {
            set_status_text(runner->ui_state, "No command running");
        }
        return;
    }
    runner->cancel_requested = true;
    pid_to_kill = runner->pid;
    SDL_UnlockMutex(runner->mutex);

    if (pid_to_kill > 0) {
        kill(pid_to_kill, SIGTERM);
        if (runner->ui_state) {
            append_log_line(runner->ui_state, "[CMD] Sent SIGTERM to running command");
            set_status_text(runner->ui_state, "Stopping command...");
        }
    }
}

static void draw_text(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color, int x, int y, const char *text) {
    if (!font || !text) {
        return;
    }
    SDL_Surface *surface = TTF_RenderUTF8_Blended(font, text, color);
    if (!surface) {
        return;
    }
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    int w = surface->w;
    int h = surface->h;
    SDL_FreeSurface(surface);
    if (!texture) {
        return;
    }
    SDL_Rect dst = {x, y, w, h};
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

static void draw_text_centered(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color, const SDL_Rect *rect, const char *text) {
    if (!font || !rect || !text) {
        return;
    }
    SDL_Surface *surface = TTF_RenderUTF8_Blended(font, text, color);
    if (!surface) {
        return;
    }
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    int w = surface->w;
    int h = surface->h;
    SDL_FreeSurface(surface);
    if (!texture) {
        return;
    }
    SDL_Rect dst = {
        rect->x + (rect->w - w) / 2,
        rect->y + (rect->h - h) / 2,
        w,
        h
    };
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

static int font_line_height(TTF_Font *font) {
    int h = font ? TTF_FontLineSkip(font) : 18;
    if (h <= 0) {
        h = 18;
    }
    return h;
}

static void render_buttons(SDL_Renderer *renderer, TTF_Font *font, const button_t *buttons, size_t button_count, action_t active_action) {
    SDL_Color text_color = {20, 20, 30, 255};
    SDL_Color text_active = {255, 255, 255, 255};

    for (size_t i = 0; i < button_count; ++i) {
        const button_t *button = &buttons[i];
        bool active = (button->action != ACTION_STOP_COMMAND && button->action == active_action);

        SDL_Color fill = active ? (SDL_Color){50, 150, 255, 255} : (SDL_Color){200, 210, 230, 255};
        SDL_SetRenderDrawColor(renderer, fill.r, fill.g, fill.b, fill.a);
        SDL_RenderFillRect(renderer, &button->rect);

        SDL_SetRenderDrawColor(renderer, 30, 40, 80, 255);
        SDL_RenderDrawRect(renderer, &button->rect);

        draw_text_centered(renderer, font, active ? text_active : text_color, &button->rect, button->label);
    }
}

static void render_telemetry(SDL_Renderer *renderer, TTF_Font *font, const telemetry_snapshot_t *snapshot) {
    SDL_Color label_color = {220, 230, 255, 255};
    SDL_Color bar_bg = {18, 32, 64, 255};
    SDL_Color bar_fg = {45, 160, 255, 255};

    char header[128];
    snprintf(header, sizeof(header), "Telemetry — Tick %d (%s)", snapshot->tick, snapshot->mode[0] ? snapshot->mode : "mode?");
    draw_text(renderer, font, label_color, 20, 90, header);

    int line_height = font_line_height(font) + 8;
    int base_y = 130;
    int bar_x = 220;
    int bar_width = 320;

    for (size_t i = 0; i < snapshot->metric_count; ++i) {
        const double value = snapshot->values[i];
        double clamped = value;
        if (clamped < 0.0) {
            clamped = 0.0;
        } else if (clamped > 1.0) {
            clamped = 1.0;
        }

        int y = base_y + (int)i * line_height;
        SDL_Rect bg_rect = {bar_x, y, bar_width, 20};
        SDL_SetRenderDrawColor(renderer, bar_bg.r, bar_bg.g, bar_bg.b, bar_bg.a);
        SDL_RenderFillRect(renderer, &bg_rect);

        SDL_Rect fg_rect = {bar_x, y, (int)((double)bar_width * clamped), 20};
        SDL_SetRenderDrawColor(renderer, bar_fg.r, bar_fg.g, bar_fg.b, bar_fg.a);
        SDL_RenderFillRect(renderer, &fg_rect);

        char label[128];
        snprintf(label, sizeof(label), "%s", snapshot->names[i]);
        draw_text(renderer, font, label_color, 20, y - 2, label);

        char value_text[64];
        snprintf(value_text, sizeof(value_text), "%.4f", value);
        draw_text(renderer, font, label_color, bar_x + bar_width + 10, y - 2, value_text);
    }
}

static void render_pocket_metrics(SDL_Renderer *renderer, TTF_Font *font, const pocket_metrics_t *metrics) {
    SDL_Color label_color = {240, 240, 250, 255};
    int x = 20;
    int y = 450;
    int line_height = font_line_height(font) + 2;

    draw_text(renderer, font, label_color, x, y, "Pocket Dimension Metrics");
    y += line_height;

    char line[128];
    snprintf(line, sizeof(line), "Tick: %d", metrics->tick);
    draw_text(renderer, font, label_color, x, y, line);
    y += line_height;

    snprintf(line, sizeof(line), "Average Score: %.4f", metrics->average_score);
    draw_text(renderer, font, label_color, x, y, line);
    y += line_height;

    snprintf(line, sizeof(line), "Average Coherence: %.4f", metrics->average_coherence);
    draw_text(renderer, font, label_color, x, y, line);
    y += line_height;

    snprintf(line, sizeof(line), "Average Decoherence: %.4f", metrics->average_decoherence);
    draw_text(renderer, font, label_color, x, y, line);
    y += line_height;

    snprintf(line, sizeof(line), "Memory Usage: %.2f MB", metrics->memory_usage_mb);
    draw_text(renderer, font, label_color, x, y, line);
    y += line_height;

    snprintf(line, sizeof(line), "Memory Peak: %.2f MB", metrics->memory_peak_mb);
    draw_text(renderer, font, label_color, x, y, line);
}

static void render_log(SDL_Renderer *renderer, TTF_Font *font, ui_state_t *state) {
    SDL_Rect log_rect = {520, 90, WINDOW_WIDTH - 540, WINDOW_HEIGHT - 200};
    SDL_SetRenderDrawColor(renderer, 18, 26, 50, 240);
    SDL_RenderFillRect(renderer, &log_rect);
    SDL_SetRenderDrawColor(renderer, 40, 60, 100, 255);
    SDL_RenderDrawRect(renderer, &log_rect);

    SDL_Color text_color = {220, 225, 240, 255};
    int line_height = font_line_height(font);
    int max_lines = log_rect.h / line_height;

    SDL_LockMutex(state->mutex);
    int available = state->log_size < max_lines ? state->log_size : max_lines;
    int start = (state->log_start + state->log_size - available + MAX_LOG_LINES) % MAX_LOG_LINES;
    for (int i = 0; i < available; ++i) {
        int index = (start + i) % MAX_LOG_LINES;
        const char *line = state->log_lines[index];
        draw_text(renderer, font, text_color, log_rect.x + 8, log_rect.y + 4 + i * line_height, line);
    }
    SDL_UnlockMutex(state->mutex);

    draw_text(renderer, font, text_color, log_rect.x + 8, log_rect.y - font_line_height(font), "Command Output");
}

static void render_status(SDL_Renderer *renderer, TTF_Font *font, ui_state_t *state) {
    SDL_Color status_color = {200, 220, 255, 255};
    char status_copy[MAX_LOG_LEN];

    SDL_LockMutex(state->mutex);
    strncpy(status_copy, state->status_text, sizeof(status_copy) - 1);
    status_copy[sizeof(status_copy) - 1] = '\0';
    SDL_UnlockMutex(state->mutex);

    draw_text(renderer, font, status_color, 20, WINDOW_HEIGHT - 60, status_copy);

    SDL_Color hint_color = {180, 190, 220, 255};
    draw_text(renderer, font, hint_color, 20, WINDOW_HEIGHT - 30, "Keys: [B] Build CUDA  [R] Run Binary  [A] Run Accelerator  [1-3] Phases  [S] Stop  [Esc] Quit");
}

static void render_ui(SDL_Renderer *renderer, TTF_Font *font, ui_state_t *state, const button_t *buttons, size_t button_count, action_t active_action) {
    SDL_SetRenderDrawColor(renderer, 16, 22, 48, 255);
    SDL_RenderClear(renderer);

    SDL_Color title_color = {210, 225, 255, 255};
    draw_text(renderer, font, title_color, 20, 20, "Qallow Control & Telemetry");

    render_buttons(renderer, font, buttons, button_count, active_action);
    render_telemetry(renderer, font, &state->telemetry);
    render_pocket_metrics(renderer, font, &state->pocket);
    render_log(renderer, font, state);
    render_status(renderer, font, state);

    SDL_RenderPresent(renderer);
}

static void build_command_request(command_request_t *request, const char *label, const char *working_dir, int argc, const char *argv[]) {
    memset(request, 0, sizeof(*request));
    strncpy(request->label, label, sizeof(request->label) - 1);
    if (working_dir) {
        strncpy(request->working_dir, working_dir, sizeof(request->working_dir) - 1);
    }
    request->argc = argc;
    for (int i = 0; i < argc && i < MAX_CMD_ARGS; ++i) {
        strncpy(request->argv[i], argv[i], sizeof(request->argv[i]) - 1);
    }
}

static bool trigger_action(action_t action, const ui_config_t *cfg, ui_state_t *state, command_runner_t *runner) {
    command_request_t request;
    const char *argv[MAX_CMD_ARGS];

    switch (action) {
        case ACTION_BUILD_CUDA: {
            const char *script = "scripts/build_unified_cuda.sh";
            if (access(script, X_OK) != 0) {
                const char *alt = "scripts/build_unified_cuda.sh";
                if (access(alt, F_OK) != 0) {
                    append_log_line(state, "[ERROR] scripts/build_unified_cuda.sh not found");
                    set_status_text(state, "Build script not found");
                    return false;
                }
            }
            argv[0] = "bash";
            argv[1] = "scripts/build_unified_cuda.sh";
            build_command_request(&request, "Building CUDA pipeline…", cfg->repo_root, 2, argv);
            break;
        }
        case ACTION_RUN_BINARY: {
            if (cfg->runner_path[0] == '\0') {
                append_log_line(state, "[ERROR] Runner binary not found");
                set_status_text(state, "Runner binary not found");
                return false;
            }
            argv[0] = cfg->runner_path;
            build_command_request(&request, "Running Qallow binary…", cfg->repo_root, 1, argv);
            break;
        }
        case ACTION_RUN_ACCELERATOR: {
            if (access("scripts/run_auto.sh", F_OK) != 0) {
                append_log_line(state, "[ERROR] scripts/run_auto.sh not found");
                set_status_text(state, "Accelerator script not found");
                return false;
            }
            argv[0] = "bash";
            argv[1] = "scripts/run_auto.sh";
            argv[2] = "--watch";
            argv[3] = cfg->repo_root[0] ? cfg->repo_root : ".";
            build_command_request(&request, "Running accelerator…", cfg->repo_root, 4, argv);
            break;
        }
        case ACTION_PHASE_14:
        case ACTION_PHASE_15:
        case ACTION_PHASE_16: {
            if (cfg->runner_path[0] == '\0') {
                append_log_line(state, "[ERROR] Runner binary not found");
                set_status_text(state, "Runner binary not found");
                return false;
            }
            int phase = action == ACTION_PHASE_14 ? 14 : (action == ACTION_PHASE_15 ? 15 : 16);
            char phase_arg[32];
            snprintf(phase_arg, sizeof(phase_arg), "--phase=%d", phase);
            argv[0] = cfg->runner_path;
            argv[1] = phase_arg;
            char label[64];
            snprintf(label, sizeof(label), "Executing Phase %d…", phase);
            build_command_request(&request, label, cfg->repo_root, 2, argv);
            break;
        }
        case ACTION_STOP_COMMAND:
            command_runner_stop(runner);
            return true;
        case ACTION_NONE:
        default:
            return false;
    }

    return command_runner_start(runner, &request, action);
}

static void setup_buttons(button_t *buttons, size_t *count) {
    const int button_width = 150;
    const int button_height = 42;
    const int spacing = 12;
    const int top = 50;
    int x = 20;

    const struct {
        action_t action;
        const char *label;
    } defs[] = {
        {ACTION_BUILD_CUDA, "Build CUDA"},
        {ACTION_RUN_BINARY, "Run Binary"},
        {ACTION_RUN_ACCELERATOR, "Run Accelerator"},
        {ACTION_PHASE_14, "Phase 14"},
        {ACTION_PHASE_15, "Phase 15"},
        {ACTION_PHASE_16, "Phase 16"},
        {ACTION_STOP_COMMAND, "Stop"}
    };

    size_t idx = 0;
    for (size_t i = 0; i < sizeof(defs) / sizeof(defs[0]) && idx < MAX_BUTTONS; ++i) {
        buttons[idx].action = defs[i].action;
        buttons[idx].label = defs[i].label;
        buttons[idx].rect.x = x;
        buttons[idx].rect.y = top;
        buttons[idx].rect.w = button_width;
        buttons[idx].rect.h = button_height;
        x += button_width + spacing;
        ++idx;
    }
    *count = idx;
}

static action_t hit_test_buttons(const button_t *buttons, size_t count, int x, int y) {
    for (size_t i = 0; i < count; ++i) {
        const SDL_Rect *r = &buttons[i].rect;
        if (x >= r->x && x <= r->x + r->w && y >= r->y && y <= r->y + r->h) {
            return buttons[i].action;
        }
    }
    return ACTION_NONE;
}

static action_t snapshot_active_action(command_runner_t *runner) {
    action_t action = ACTION_NONE;
    SDL_LockMutex(runner->mutex);
    if (runner->active) {
        action = runner->active_action;
    }
    SDL_UnlockMutex(runner->mutex);
    return action;
}

int main(int argc, char **argv) {
    ui_config_t config;
    parse_arguments(argc, argv, &config);

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
        fprintf(stderr, "SDL initialization failed: %s\n", SDL_GetError());
        return EXIT_FAILURE;
    }

    if (TTF_Init() < 0) {
        fprintf(stderr, "SDL_ttf initialization failed: %s\n", TTF_GetError());
        SDL_Quit();
        return EXIT_FAILURE;
    }

    TTF_Font *font = NULL;
    if (config.font_path) {
        font = TTF_OpenFont(config.font_path, 18);
        if (!font) {
            fprintf(stderr, "Failed to load font '%s': %s\n", config.font_path, TTF_GetError());
        }
    }

    SDL_Window *window = SDL_CreateWindow(
        "Qallow Control Center",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );

    if (!window) {
        fprintf(stderr, "Failed to create window: %s\n", SDL_GetError());
        if (font) {
            TTF_CloseFont(font);
        }
        TTF_Quit();
        SDL_Quit();
        return EXIT_FAILURE;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Failed to create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        if (font) {
            TTF_CloseFont(font);
        }
        TTF_Quit();
        SDL_Quit();
        return EXIT_FAILURE;
    }

    ui_state_t state;
    ui_state_init(&state);

    command_runner_t runner;
    command_runner_init(&runner, &state);

    button_t buttons[MAX_BUTTONS];
    size_t button_count = 0;
    setup_buttons(buttons, &button_count);

    read_latest_snapshot(config.telemetry_path, &state.telemetry);
    read_pocket_metrics(config.pocket_metrics_path, &state.pocket);

    Uint32 last_refresh = 0;
    Uint32 last_pocket_refresh = 0;

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_KEYDOWN) {
                SDL_Keycode key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE) {
                    running = false;
                } else if (key == SDLK_b) {
                    trigger_action(ACTION_BUILD_CUDA, &config, &state, &runner);
                } else if (key == SDLK_r) {
                    trigger_action(ACTION_RUN_BINARY, &config, &state, &runner);
                } else if (key == SDLK_a) {
                    trigger_action(ACTION_RUN_ACCELERATOR, &config, &state, &runner);
                } else if (key == SDLK_1) {
                    trigger_action(ACTION_PHASE_14, &config, &state, &runner);
                } else if (key == SDLK_2) {
                    trigger_action(ACTION_PHASE_15, &config, &state, &runner);
                } else if (key == SDLK_3) {
                    trigger_action(ACTION_PHASE_16, &config, &state, &runner);
                } else if (key == SDLK_s) {
                    trigger_action(ACTION_STOP_COMMAND, &config, &state, &runner);
                }
            } else if (event.type == SDL_MOUSEBUTTONDOWN) {
                if (event.button.button == SDL_BUTTON_LEFT) {
                    action_t action = hit_test_buttons(buttons, button_count, event.button.x, event.button.y);
                    if (action != ACTION_NONE) {
                        trigger_action(action, &config, &state, &runner);
                    }
                }
            }
        }

        Uint32 now = SDL_GetTicks();
        if (now - last_refresh >= config.refresh_interval_ms) {
            read_latest_snapshot(config.telemetry_path, &state.telemetry);
            last_refresh = now;
        }
        if (now - last_pocket_refresh >= config.pocket_refresh_ms) {
            read_pocket_metrics(config.pocket_metrics_path, &state.pocket);
            last_pocket_refresh = now;
        }

        action_t active_action = snapshot_active_action(&runner);
        render_ui(renderer, font, &state, buttons, button_count, active_action);
        SDL_Delay(16);
    }

    command_runner_destroy(&runner);
    ui_state_destroy(&state);

    if (font) {
        TTF_CloseFont(font);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return EXIT_SUCCESS;
}
