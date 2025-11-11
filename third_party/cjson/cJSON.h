/*
  Copyright (c) 2009-2017 Dave Gamble and cJSON contributors

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
  THE SOFTWARE.
*/

#ifndef cJSON__h
#define cJSON__h

#ifdef __cplusplus
extern "C"
{
#endif

#if !defined(__WINDOWS__) && (defined(WIN32) || defined(WIN64) || defined(_MSC_VER) || defined(_WIN32))
#define __WINDOWS__
#endif

#ifdef __WINDOWS__

/* When compiling for windows, we specify a specific calling convention to avoid issues where we are being called from a project with a different default calling convention.  For windows you have 3 define options:

CJSON_HIDE_SYMBOLS - Define this in the case where you don't want to ever export symbols
CJSON_EXPORT_SYMBOLS - Define this on library build when you want to export symbols (and also define CJSON_API_VISIBILITY)
CJSON_IMPORT_SYMBOLS - Define this on application build when you want to import symbols

For GCC, options are handled automatically by the visibility attribute.
*/

/* setup cJSON_API */
#if defined(CJSON_HIDE_SYMBOLS)
    #define cJSON_API
#elif defined(CJSON_EXPORT_SYMBOLS)
    #define cJSON_API __declspec(dllexport)
#elif defined(CJSON_IMPORT_SYMBOLS)
    #define cJSON_API __declspec(dllimport)
#else
    #define cJSON_API
#endif

/* setup cJSON_AS_INLINE */
#if defined(_MSC_VER)
    #define cJSON_AS_INLINE __inline
#else
    #define cJSON_AS_INLINE inline
#endif

#else /* !__WINDOWS__ */

/* setup cJSON_API */
#if (defined(__GNUC__) || defined(__SUNPRO_CC) || defined (__SUNPRO_C)) && defined(CJSON_API_VISIBILITY)
    #define cJSON_API __attribute__((visibility("default")))
#else
    #define cJSON_API
#endif

/* setup cJSON_AS_INLINE */
#if defined(__GNUC__)
    #define cJSON_AS_INLINE __inline__
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
    #define cJSON_AS_INLINE inline
#else
    #define cJSON_AS_INLINE
#endif

#endif /* !__WINDOWS__ */

/* project version */
#define CJSON_VERSION_MAJOR 1
#define CJSON_VERSION_MINOR 7
#define CJSON_VERSION_PATCH 15

#include <stddef.h>

/* cJSON Types: */
#define cJSON_Invalid (0)
#define cJSON_False  (1 << 0)
#define cJSON_True   (1 << 1)
#define cJSON_NULL   (1 << 2)
#define cJSON_Number (1 << 3)
#define cJSON_String (1 << 4)
#define cJSON_Array  (1 << 5)
#define cJSON_Object (1 << 6)
#define cJSON_Raw    (1 << 7) /* raw json */

#define cJSON_IsReference 256
#define cJSON_StringIsConst 512

/* Type definition for cJSON types */
typedef int cJSON_Type;

/* The cJSON structure: */
typedef struct cJSON
{
    /* next/prev allow you to walk array/object chains. Alternatively, use GetArraySize/GetArrayItem/GetObjectItem */
    struct cJSON *next;
    struct cJSON *prev;
    /* An array or object item will have a child pointer pointing to a chain of the items in the array/object. */
    struct cJSON *child;

    /* The type of the item, as above. */
    int type;

    /* The item's string, if type==cJSON_String  and type == cJSON_Raw */
    char *valuestring;
    /* writing to valueint is DEPRECATED, use cJSON_SetNumberValue instead */
    int valueint;
    /* The item's number, if type==cJSON_Number */
    double valuedouble;

    /* The item's name string, if this item is the child of an object. */
    char *string;
} cJSON;

typedef struct cJSON_Hooks
{
      /* malloc/free are CDECL on Windows regardless of the default calling convention of the compiler, so ensure the hooks allow passing those functions directly. */
      void *(*malloc_fn)(size_t sz);
      void (*free_fn)(void *ptr);
} cJSON_Hooks;

typedef int cJSON_bool;

/* Limits how deeply nested arrays/objects can be before cJSON rejects to parse them.
 * This is to prevent stack overflows. */
#ifndef CJSON_NESTING_LIMIT
#define CJSON_NESTING_LIMIT 1000
#endif

/* Supply malloc, realloc and free functions to cJSON */
cJSON_API void cJSON_InitHooks(cJSON_Hooks* hooks);

/* Memory Management: the caller is always responsible to free the results from all variants of cJSON_Parse (with cJSON_Delete) and cJSON_Print (with stdlib free, cJSON_Hooks.free_fn, or cJSON_free as appropriate). The exception is cJSON_PrintPreallocated, where the caller provides a buffer to be filled. */
/* Supply a block of JSON, and this returns a cJSON object you can interrogate. Call cJSON_Delete when finished. */
cJSON_API cJSON *cJSON_Parse(const char *value);
cJSON_API cJSON *cJSON_ParseWithLength(const char *value, size_t buffer_length);
/* ParseWithOpts allows you to require (and check) that the JSON is null terminated, and to retrieve pointers to error messages. */
/* If you supply a ptr in return_parse_end and parsing fails, then return_parse_end will contain a pointer to the error. If not, then cJSON_GetErrorPtr() does the job. */
cJSON_API cJSON *cJSON_ParseWithOpts(const char *value, const char **return_parse_end, cJSON_bool require_null_terminated);
cJSON_API cJSON *cJSON_ParseWithLengthOpts(const char *value, size_t buffer_length, const char **return_parse_end, cJSON_bool require_null_terminated);

/* Render a cJSON entity to text for transfer/storage. Free the char* when finished. */
cJSON_API char *cJSON_Print(const cJSON *item);
/* Render a cJSON entity to text for transfer/storage without any formatting. Free the char* when finished. */
cJSON_API char *cJSON_PrintUnformatted(const cJSON *item);
/* Render a cJSON entity to text using a buffered strategy. prebuffer is a guess at the final size. guessing well reduces reallocation. fmt=0 gives unformatted, =1 gives formatted */
cJSON_API char *cJSON_PrintBuffered(const cJSON *item, int prebuffer, cJSON_bool fmt);
/* Render a cJSON entity to text directly into a customer provided buffer. */
/* buffer_length is the size of the buffer. If it's not long enough, it will return cJSON_False and fill buffer[0] with 0. */
/* Otherwise, it will return cJSON_True. */
cJSON_API cJSON_bool cJSON_PrintPreallocated(cJSON *item, char *buffer, const int length, const cJSON_bool format);
/* Delete a cJSON entity and all subentities. */
cJSON_API void cJSON_Delete(cJSON *item);

/* Returns the number of items in an array (or object). */
cJSON_API int cJSON_GetArraySize(const cJSON *array);
/* Retrieve item number "index" from array "array". Returns NULL if unsuccessful. */
cJSON_API cJSON *cJSON_GetArrayItem(const cJSON *array, int index);
/* Get item "string" from object. Case sensitive. */
cJSON_API cJSON *cJSON_GetObjectItem(const cJSON * const object, const char * const string);
cJSON_API cJSON *cJSON_GetObjectItemCaseSensitive(const cJSON * const object, const char * const string);
cJSON_API cJSON_bool cJSON_HasObjectItem(const cJSON *object, const char *string);
/* For analysing failed parses. This returns a pointer to the parse error. You'll probably need to look a few chars back to make sense of it. Defined when cJSON_Parse() returns 0. 0 when cJSON_Parse() succeeds. */
cJSON_API const char *cJSON_GetErrorPtr(void);

/* Check item type */
cJSON_API cJSON_bool cJSON_IsInvalid(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsFalse(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsTrue(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsBool(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsNull(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsNumber(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsString(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsArray(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsObject(const cJSON * const item);
cJSON_API cJSON_bool cJSON_IsRaw(const cJSON * const item);

/* These functions check the type of an item */
#define cJSON_IsRef(item) (((item) != NULL) && ((item)->type & cJSON_IsReference))
#define cJSON_StringIsCnst(item) (((item) != NULL) && ((item)->type & cJSON_StringIsConst))

/* Create basic types: */
cJSON_API cJSON *cJSON_CreateNull(void);
cJSON_API cJSON *cJSON_CreateTrue(void);
cJSON_API cJSON *cJSON_CreateFalse(void);
cJSON_API cJSON *cJSON_CreateBool(cJSON_bool boolean);
cJSON_API cJSON *cJSON_CreateNumber(double num);
cJSON_API cJSON *cJSON_CreateString(const char *string);
/* raw json */
cJSON_API cJSON *cJSON_CreateRaw(const char *raw);
cJSON_API cJSON *cJSON_CreateArray(void);
cJSON_API cJSON *cJSON_CreateObject(void);

/* Create Arrays: */
cJSON_API cJSON *cJSON_CreateIntArray(const int *numbers, int count);
cJSON_API cJSON *cJSON_CreateFloatArray(const float *numbers, int count);
cJSON_API cJSON *cJSON_CreateDoubleArray(const double *numbers, int count);
cJSON_API cJSON *cJSON_CreateStringArray(const char *const *strings, int count);

/* Helpers for extracting primitive values without manual type checks. */
cJSON_API char *cJSON_GetStringValue(const cJSON * const item);
cJSON_API double cJSON_GetNumberValue(const cJSON * const item);

/* Append item to the specified array/object. */
cJSON_API void cJSON_AddItemToArray(cJSON *array, cJSON *item);
cJSON_API void cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item);
/* Use this when string is definitely const (i.e. a literal, or as good as), and will definitely survive the cJSON object.
 * WARNING: When this function was used, make sure to use cJSON_DeleteWithConstKeys afterwards to avoid freeing memory that was not allocated by cJSON. */
cJSON_API void cJSON_AddItemToObjectCS(cJSON *object, const char *string, cJSON *item);
/* Append reference to item to the specified array/object. Use this when you want to add an existing cJSON to a new cJSON, but don't want to corrupt either. */
cJSON_API void cJSON_AddItemReferenceToArray(cJSON *array, cJSON *item);
cJSON_API void cJSON_AddItemReferenceToObject(cJSON *object, const char *string, cJSON *item);

/* Remove/Detach items from Arrays/Objects. */
cJSON_API cJSON *cJSON_DetachItemViaPointer(cJSON *parent, cJSON * const item);
cJSON_API cJSON *cJSON_DetachItemFromArray(cJSON *array, int which);
cJSON_API void cJSON_DeleteItemFromArray(cJSON *array, int which);
cJSON_API cJSON *cJSON_DetachItemFromObject(cJSON *object, const char *string);
cJSON_API cJSON *cJSON_DetachItemFromObjectCaseSensitive(cJSON *object, const char *string);
cJSON_API void cJSON_DeleteItemFromObject(cJSON *object, const char *string);
cJSON_API void cJSON_DeleteItemFromObjectCaseSensitive(cJSON *object, const char *string);

/* Update array items. */
cJSON_API void cJSON_InsertItemInArray(cJSON *array, int which, cJSON *newitem); /* Shifts pre-existing items to the right. */
cJSON_API cJSON_bool cJSON_ReplaceItemViaPointer(cJSON * const parent, cJSON * const item, cJSON * replacement);
cJSON_API void cJSON_ReplaceItemInArray(cJSON *array, int which, cJSON *newitem);
cJSON_API void cJSON_ReplaceItemInObject(cJSON *object,const char *string,cJSON *newitem);
cJSON_API void cJSON_ReplaceItemInObjectCaseSensitive(cJSON *object,const char *string,cJSON *newitem);

/* Duplicate a cJSON item */
cJSON_API cJSON *cJSON_Duplicate(const cJSON *item, cJSON_bool recurse);

/* Compare two cJSON items for equality. */
/* Returns cJSON_True if items are equal. */
cJSON_API cJSON_bool cJSON_Compare(const cJSON * const a, const cJSON * const b, const cJSON_bool case_sensitive);

/* Minify a strings, remove blank characters(such as ' ', '\t', '\r', '\n') from strings.
 * The input pointer is updated to point to the end of the last processed character. */
cJSON_API void cJSON_Minify(char *json);

/* Helper functions for creating and adding items to an object. */
/* Appends a new otherwise uninitialized item to the end of an object. */
cJSON_API cJSON* cJSON_AddNullToObject(cJSON * const object, const char * const name);
cJSON_API cJSON* cJSON_AddTrueToObject(cJSON * const object, const char * const name);
cJSON_API cJSON* cJSON_AddFalseToObject(cJSON * const object, const char * const name);
cJSON_API cJSON* cJSON_AddBoolToObject(cJSON * const object, const char * const name, const cJSON_bool boolean);
cJSON_API cJSON* cJSON_AddNumberToObject(cJSON * const object, const char * const name, const double number);
cJSON_API cJSON* cJSON_AddStringToObject(cJSON * const object, const char * const name, const char * const string);
cJSON_API cJSON* cJSON_AddRawToObject(cJSON * const object, const char * const name, const char * const raw);
cJSON_API cJSON* cJSON_AddObjectToObject(cJSON * const object, const char * const name);
cJSON_API cJSON* cJSON_AddArrayToObject(cJSON * const object, const char * const name);

/* When assigning an integer value, it needs to be propagated to valuedouble too. */
#define cJSON_SetIntValue(object, number) ((object) ? (object)->valueint = (object)->valuedouble = (number) : (number))
/* helper for the cJSON_SetNumberValue macro */
cJSON_API double cJSON_SetNumberHelper(cJSON *object, double number);
#define cJSON_SetNumberValue(object, number) ((object != NULL) ? cJSON_SetNumberHelper(object, (double)number) : (number))
/* Change the valuestring of a cJSON_String item, only if it is a string type. */
cJSON_API char* cJSON_SetValuestring(cJSON *object, const char *valuestring);

/* Macro for iterating over an array */
#define cJSON_ArrayForEach(element, array) for(element = (array != NULL) ? (array)->child : NULL; element != NULL; element = element->next)

/* malloc/free objects using the malloc/free functions that have been set with cJSON_InitHooks */
cJSON_API void *cJSON_malloc(size_t size);
cJSON_API void cJSON_free(void *object);

#ifdef __cplusplus
}
#endif

#endif
