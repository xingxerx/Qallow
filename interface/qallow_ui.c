#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define WINDOW_WIDTH 960
#define WINDOW_HEIGHT 600

#define MAX_METRICS 16
#define TOKEN_LEN 64

typedef struct {
    char names[MAX_METRICS][TOKEN_LEN];
    double values[MAX_METRICS];
    size_t metric_count;
    int tick;
    char mode[TOKEN_LEN];
} telemetry_snapshot_t;

typedef struct {
    const char *telemetry_path;
    const char *runner_path;
    const char *font_path;
    Uint32 refresh_interval_ms;
} ui_config_t;

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
        fprintf(stderr, "[UI] Unable to open telemetry file '%s': %s\n", path, strerror(errno));
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
        if (line[0] == '\n' || line[0] == '\0') {
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

static const char *detect_runner_binary(void) {
    if (access("./build/qallow_unified_cuda", X_OK) == 0) {
        return "./build/qallow_unified_cuda";
    }
    if (access("./build/qallow_unified", X_OK) == 0) {
        return "./build/qallow_unified";
    }
    if (access("./qallow_unified_cuda", X_OK) == 0) {
        return "./qallow_unified_cuda";
    }
    if (access("./qallow_unified", X_OK) == 0) {
        return "./qallow_unified";
    }
    return NULL;
}

static void parse_arguments(int argc, char **argv, ui_config_t *cfg) {
    cfg->telemetry_path = "data/logs/telemetry_stream.csv";
    cfg->runner_path = detect_runner_binary();
    cfg->font_path = "/usr/share/fonts/TTF/DejaVuSans.ttf";
    cfg->refresh_interval_ms = 750;

    for (int i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (strncmp(arg, "--telemetry=", 12) == 0) {
            cfg->telemetry_path = arg + 12;
        } else if (strncmp(arg, "--runner=", 9) == 0) {
            cfg->runner_path = arg + 9;
        } else if (strncmp(arg, "--font=", 7) == 0) {
            cfg->font_path = arg + 7;
        } else if (strncmp(arg, "--refresh-ms=", 13) == 0) {
            int value = atoi(arg + 13);
            if (value >= 100) {
                cfg->refresh_interval_ms = (Uint32)value;
            }
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            printf("Qallow Telemetry UI\n\n");
            printf("Options:\n");
            printf("  --telemetry=<path>   Path to telemetry CSV (default data/logs/telemetry_stream.csv)\n");
            printf("  --runner=<path>      Path to qallow executable for phase control\n");
            printf("  --font=<path>        Path to TTF font for rendering (default DejaVuSans)\n");
            printf("  --refresh-ms=<n>     Telemetry refresh interval in milliseconds (>=100)\n");
            printf("  --help, -h           Show this message\n");
            exit(0);
        }
    }
}

static int launch_phase(const char *runner, int phase, char *message, size_t message_len) {
    if (!runner) {
        snprintf(message, message_len, "Phase %d launch failed: binary not set", phase);
        return -1;
    }

    char command[256];
    snprintf(command, sizeof(command), "%s --phase=%d", runner, phase);

    int rc = system(command);
    if (rc == 0) {
        snprintf(message, message_len, "Phase %d completed successfully", phase);
    } else {
        snprintf(message, message_len, "Phase %d failed (exit code %d)", phase, rc);
    }
    return rc;
}

static void render_text(SDL_Renderer *renderer, TTF_Font *font, const char *text, int x, int y, SDL_Color color) {
    if (!font || !text) {
        return;
    }

    SDL_Surface *surface = TTF_RenderUTF8_Blended(font, text, color);
    if (!surface) {
        return;
    }

    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);
    SDL_FreeSurface(surface);
    if (!texture) {
        return;
    }

    SDL_Rect dst = {x, y, 0, 0};
    SDL_QueryTexture(texture, NULL, NULL, &dst.w, &dst.h);
    SDL_RenderCopy(renderer, texture, NULL, &dst);
    SDL_DestroyTexture(texture);
}

static void render_snapshot(SDL_Renderer *renderer, TTF_Font *font, const telemetry_snapshot_t *snapshot, const char *status_message) {
    SDL_SetRenderDrawColor(renderer, 16, 22, 48, 255);
    SDL_RenderClear(renderer);

    SDL_Color header_color = {200, 220, 255, 255};
    SDL_Color text_color = {240, 240, 240, 255};
    SDL_Color bar_color = {45, 160, 255, 255};
    SDL_Color bar_bg_color = {18, 32, 64, 255};
    SDL_Color phase_color = {180, 220, 255, 255};

    char header[128];
    snprintf(header, sizeof(header), "Qallow Telemetry — Tick %d (%s)", snapshot->tick, snapshot->mode[0] ? snapshot->mode : "mode?");
    render_text(renderer, font, header, 20, 20, header_color);

    int base_y = 70;
    int line_height = 40;
    int bar_width = WINDOW_WIDTH - 220;

    for (size_t i = 0; i < snapshot->metric_count; ++i) {
        const double value = snapshot->values[i];
        double clamped = value;
        if (clamped < 0.0) {
            clamped = 0.0;
        } else if (clamped > 1.0) {
            clamped = 1.0;
        }

        int bar_x = 200;
        int bar_y = base_y + (int)i * line_height;
        int filled = (int)((double)bar_width * clamped);

        SDL_Rect bg_rect = {bar_x, bar_y, bar_width, 24};
        SDL_SetRenderDrawColor(renderer, bar_bg_color.r, bar_bg_color.g, bar_bg_color.b, bar_bg_color.a);
        SDL_RenderFillRect(renderer, &bg_rect);

        SDL_Rect fg_rect = {bar_x, bar_y, filled, 24};
        SDL_SetRenderDrawColor(renderer, bar_color.r, bar_color.g, bar_color.b, bar_color.a);
        SDL_RenderFillRect(renderer, &fg_rect);

        char label[128];
        snprintf(label, sizeof(label), "%s: %.4f", snapshot->names[i], value);
        render_text(renderer, font, label, 20, bar_y, text_color);
    }

    render_text(renderer, font, "Controls: [1] Phase14  [2] Phase15  [3] Phase16  [Esc] Quit", 20, WINDOW_HEIGHT - 80, phase_color);
    if (status_message && *status_message) {
        render_text(renderer, font, status_message, 20, WINDOW_HEIGHT - 50, phase_color);
    }

    SDL_RenderPresent(renderer);
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
        "Qallow Telemetry",
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

    telemetry_snapshot_t snapshot = {0};
    char status_message[128] = {0};
    Uint32 last_refresh = 0;

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
                } else if (key == SDLK_1) {
                    launch_phase(config.runner_path, 14, status_message, sizeof(status_message));
                } else if (key == SDLK_2) {
                    launch_phase(config.runner_path, 15, status_message, sizeof(status_message));
                } else if (key == SDLK_3) {
                    launch_phase(config.runner_path, 16, status_message, sizeof(status_message));
                }
            }
        }

        Uint32 now = SDL_GetTicks();
        if (now - last_refresh >= config.refresh_interval_ms) {
            if (read_latest_snapshot(config.telemetry_path, &snapshot)) {
                last_refresh = now;
            }
        }

        render_snapshot(renderer, font, &snapshot, status_message);
        SDL_Delay(16);
    }

    if (font) {
        TTF_CloseFont(font);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return EXIT_SUCCESS;
}

