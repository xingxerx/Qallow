#ifndef CJSON_H
#define CJSON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    cJSON_False = 0,
    cJSON_True = 1,
    cJSON_NULL = 2,
    cJSON_Number = 3,
    cJSON_String = 4,
    cJSON_Array = 5,
    cJSON_Object = 6
} cJSON_Type;

typedef struct cJSON {
    struct cJSON* next;
    struct cJSON* prev;
    struct cJSON* child;
    cJSON_Type type;
    char* valuestring;
    double valuedouble;
    char* string;
} cJSON;

cJSON* cJSON_CreateObject(void);
cJSON* cJSON_CreateArray(void);
cJSON* cJSON_CreateString(const char* string);
cJSON* cJSON_CreateNumber(double number);
cJSON* cJSON_CreateBool(int bool_value);
cJSON* cJSON_CreateNull(void);

void cJSON_AddItemToObject(cJSON* object, const char* string, cJSON* item);
void cJSON_AddItemToArray(cJSON* array, cJSON* item);
void cJSON_AddStringToObject(cJSON* object, const char* name, const char* string);
void cJSON_AddNumberToObject(cJSON* object, const char* name, double number);
void cJSON_AddBoolToObject(cJSON* object, const char* name, int bool_value);
void cJSON_AddNullToObject(cJSON* object, const char* name);

char* cJSON_PrintUnformatted(const cJSON* item);
char* cJSON_Print(const cJSON* item);
void cJSON_Delete(cJSON* item);

#ifdef __cplusplus
}
#endif

#endif /* CJSON_H */
