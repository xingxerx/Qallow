#include "cJSON.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static cJSON* cjson_new_item(cJSON_Type type) {
    cJSON* item = (cJSON*)calloc(1, sizeof(cJSON));
    if (!item) {
        return NULL;
    }
    item->type = type;
    return item;
}

static char* cjson_strdup(const char* string) {
    if (!string) {
        return NULL;
    }
    size_t len = strlen(string);
    char* copy = (char*)malloc(len + 1);
    if (!copy) {
        return NULL;
    }
    memcpy(copy, string, len + 1);
    return copy;
}

cJSON* cJSON_CreateObject(void) {
    return cjson_new_item(cJSON_Object);
}

cJSON* cJSON_CreateArray(void) {
    return cjson_new_item(cJSON_Array);
}

cJSON* cJSON_CreateString(const char* string) {
    cJSON* item = cjson_new_item(cJSON_String);
    if (!item) {
        return NULL;
    }
    item->valuestring = cjson_strdup(string ? string : "");
    if (!item->valuestring) {
        free(item);
        return NULL;
    }
    return item;
}

cJSON* cJSON_CreateNumber(double number) {
    cJSON* item = cjson_new_item(cJSON_Number);
    if (!item) {
        return NULL;
    }
    item->valuedouble = number;
    return item;
}

cJSON* cJSON_CreateBool(int bool_value) {
    return cjson_new_item(bool_value ? cJSON_True : cJSON_False);
}

cJSON* cJSON_CreateNull(void) {
    return cjson_new_item(cJSON_NULL);
}

static void cjson_append_child(cJSON* parent, cJSON* item) {
    if (!parent || !item) {
        return;
    }
    if (!parent->child) {
        parent->child = item;
    } else {
        cJSON* sibling = parent->child;
        while (sibling->next) {
            sibling = sibling->next;
        }
        sibling->next = item;
        item->prev = sibling;
    }
}

void cJSON_AddItemToObject(cJSON* object, const char* string, cJSON* item) {
    if (!object || object->type != cJSON_Object || !item) {
        return;
    }
    if (item->string) {
        free(item->string);
    }
    item->string = cjson_strdup(string ? string : "");
    cjson_append_child(object, item);
}

void cJSON_AddItemToArray(cJSON* array, cJSON* item) {
    if (!array || array->type != cJSON_Array || !item) {
        return;
    }
    cjson_append_child(array, item);
}

cJSON* cJSON_AddStringToObject(cJSON* const object, const char* const name, const char* const string) {
    cJSON* item = cJSON_CreateString(string);
    cJSON_AddItemToObject(object, name, item);
    return item;
}

cJSON* cJSON_AddNumberToObject(cJSON* const object, const char* const name, const double number) {
    cJSON* item = cJSON_CreateNumber(number);
    cJSON_AddItemToObject(object, name, item);
    return item;
}

cJSON* cJSON_AddBoolToObject(cJSON* const object, const char* const name, const cJSON_bool boolean) {
    cJSON* item = cJSON_CreateBool(boolean);
    cJSON_AddItemToObject(object, name, item);
    return item;
}

cJSON* cJSON_AddNullToObject(cJSON* const object, const char* const name) {
    cJSON* item = cJSON_CreateNull();
    cJSON_AddItemToObject(object, name, item);
    return item;
}

typedef struct {
    char* buffer;
    size_t length;
    size_t capacity;
} cjson_buffer_t;

static int cjson_buffer_reserve(cjson_buffer_t* buf, size_t extra) {
    if (buf->length + extra + 1 <= buf->capacity) {
        return 0;
    }
    size_t new_capacity = buf->capacity ? buf->capacity * 2 : 256;
    while (new_capacity < buf->length + extra + 1) {
        new_capacity *= 2;
    }
    char* new_buffer = (char*)realloc(buf->buffer, new_capacity);
    if (!new_buffer) {
        return -1;
    }
    buf->buffer = new_buffer;
    buf->capacity = new_capacity;
    return 0;
}

static int cjson_buffer_append(cjson_buffer_t* buf, const char* text, size_t len) {
    if (cjson_buffer_reserve(buf, len) != 0) {
        return -1;
    }
    memcpy(buf->buffer + buf->length, text, len);
    buf->length += len;
    buf->buffer[buf->length] = '\0';
    return 0;
}

static int cjson_buffer_append_char(cjson_buffer_t* buf, char c) {
    return cjson_buffer_append(buf, &c, 1);
}

static int cjson_print_value(const cJSON* item, cjson_buffer_t* buf);

static int cjson_print_string(const char* string, cjson_buffer_t* buf) {
    if (cjson_buffer_append_char(buf, '"') != 0) {
        return -1;
    }
    for (const char* p = string; p && *p; ++p) {
        char c /* TODO: Use more descriptive name */= *p;
        if (c == '"' || c == '\\') {
            if (cjson_buffer_append_char(buf, '\\') != 0) {
                return -1;
            }
            if (cjson_buffer_append_char(buf, c) != 0) {
                return -1;
            }
        } else if (c == '\n') {
            if (cjson_buffer_append(buf, "\\n", 2) != 0) {
                return -1;
            }
        } else {
            if (cjson_buffer_append_char(buf, c) != 0) {
                return -1;
            }
        }
    }
    return cjson_buffer_append_char(buf, '"');
}

static int cjson_print_array(const cJSON* array, cjson_buffer_t* buf) {
    if (cjson_buffer_append_char(buf, '[') != 0) {
        return -1;
    }
    const cJSON* item = array->child;
    int first = 1;
    while (item) {
        if (!first) {
            if (cjson_buffer_append_char(buf, ',') != 0) {
                return -1;
            }
        }
        if (cjson_print_value(item, buf) != 0) {
            return -1;
        }
        first = 0;
        item = item->next;
    }
    return cjson_buffer_append_char(buf, ']');
}

static int cjson_print_object(const cJSON* object, cjson_buffer_t* buf) {
    if (cjson_buffer_append_char(buf, '{') != 0) {
        return -1;
    }
    const cJSON* item = object->child;
    int first = 1;
    while (item) {
        if (!first) {
            if (cjson_buffer_append_char(buf, ',') != 0) {
                return -1;
            }
        }
        if (cjson_print_string(item->string ? item->string : "", buf) != 0) {
            return -1;
        }
        if (cjson_buffer_append_char(buf, ':') != 0) {
            return -1;
        }
        if (cjson_print_value(item, buf) != 0) {
            return -1;
        }
        first = 0;
        item = item->next;
    }
    return cjson_buffer_append_char(buf, '}');
}

static int cjson_print_value(const cJSON* item, cjson_buffer_t* buf) {
    if (!item) {
        return -1;
    }
    switch (item->type) {
        case cJSON_False:
            return cjson_buffer_append(buf, "false", 5);
        case cJSON_True:
            return cjson_buffer_append(buf, "true", 4);
        case cJSON_NULL:
            return cjson_buffer_append(buf, "null", 4);
        case cJSON_Number: {
            char number_buffer[64];
            int written = snprintf(number_buffer, sizeof(number_buffer), "%.15g", item->valuedouble);
            if (written < 0) {
                return -1;
            }
            return cjson_buffer_append(buf, number_buffer, (size_t)written);
        }
        case cJSON_String:
            return cjson_print_string(item->valuestring ? item->valuestring : "", buf);
        case cJSON_Array:
            return cjson_print_array(item, buf);
        case cJSON_Object:
            return cjson_print_object(item, buf);
        default:
            return -1;
    }
}

static char* cjson_print_internal(const cJSON* item) {
    cjson_buffer_t buf = {0};
    if (cjson_print_value(item, &buf) != 0) {
        free(buf.buffer);
        return NULL;
    }
    if (cjson_buffer_reserve(&buf, 0) != 0) {
        free(buf.buffer);
        return NULL;
    }
    return buf.buffer;
}

char* cJSON_PrintUnformatted(const cJSON* item) {
    return cjson_print_internal(item);
}

char* cJSON_Print(const cJSON* item) {
    return cjson_print_internal(item);
}

void cJSON_Delete(cJSON* item) {
    if (!item) {
        return;
    }
    cJSON* child = item->child;
    while (child) {
        cJSON* next = child->next;
        cJSON_Delete(child);
        child = next;
    }
    if (item->valuestring) {
        free(item->valuestring);
    }
    if (item->string) {
        free(item->string);
    }
    free(item);
}
