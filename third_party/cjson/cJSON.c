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

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <float.h>

#include "cJSON.h"

/* define our own boolean type */
#ifdef true
#undef true
#endif
#define true ((cJSON_bool)1)

#ifdef false
#undef false
#endif
#define false ((cJSON_bool)0)

typedef struct {
    const unsigned char *json;
    size_t position;
} error;
static error global_error = { NULL, 0 };

cJSON_API const char *cJSON_GetErrorPtr(void)
{
    return (const char*)(global_error.json + global_error.position);
}

cJSON_API char *cJSON_GetStringValue(const cJSON * const item)
{
    if (!cJSON_IsString(item))
    {
        return NULL;
    }

    return item->valuestring;
}

cJSON_API double cJSON_GetNumberValue(const cJSON * const item)
{
    if (!cJSON_IsNumber(item))
    {
        return (double) NAN;
    }

    return item->valuedouble;
}

/* This is a safeguard to prevent copy-pasters from using incompatible C and header files */
#if (CJSON_VERSION_MAJOR != 1) || (CJSON_VERSION_MINOR != 7) || (CJSON_VERSION_PATCH != 15)
    #error cJSON.h and cJSON.c have different versions. Make sure that both have the same.
#endif

cJSON_API const char* cJSON_Version(void)
{
    static char version[15];
    sprintf(version, "%i.%i.%i", CJSON_VERSION_MAJOR, CJSON_VERSION_MINOR, CJSON_VERSION_PATCH);

    return version;
}

/* Case insensitive string comparison, doesn't consider two NULL pointers equal though */
static int case_insensitive_strcmp(const unsigned char *string1, const unsigned char *string2)
{
    if ((string1 == NULL) || (string2 == NULL))
    {
        return 1;
    }

    if (string1 == string2)
    {
        return 0;
    }

    for(; tolower(*string1) == tolower(*string2); (void)string1++, string2++)
    {
        if (*string1 == '\0')
        {
            return 0;
        }
    }

    return tolower(*string1) - tolower(*string2);
}

typedef struct internal_hooks
{
    void *(*allocate)(size_t size);
    void (*deallocate)(void *pointer);
    void *(*reallocate)(void *pointer, size_t size);
} internal_hooks;

#if defined(_MSC_VER)
/* work around MSVC error C2322: '...' address of dllimport '...' is not static */
static void *internal_malloc(size_t size)
{
    return malloc(size);
}
static void internal_free(void *pointer)
{
    free(pointer);
}
static void *internal_realloc(void *pointer, size_t size)
{
    return realloc(pointer, size);
}
#else
#define internal_malloc malloc
#define internal_free free
#define internal_realloc realloc
#endif

/* strlen of character literals resolved at compile time */
#define static_strlen(string_literal) (sizeof(string_literal) - sizeof(""))

static internal_hooks global_hooks = { internal_malloc, internal_free, internal_realloc };

static unsigned char* cJSON_strdup(const unsigned char* string, const internal_hooks * const hooks)
{
    size_t length = 0;
    unsigned char *copy = NULL;

    if (string == NULL)
    {
        return NULL;
    }

    length = strlen((const char*)string) + sizeof("");
    copy = (unsigned char*)hooks->allocate(length);
    if (copy == NULL)
    {
        return NULL;
    }
    memcpy(copy, string, length);

    return copy;
}

cJSON_API void cJSON_InitHooks(cJSON_Hooks* hooks)
{
    if (hooks == NULL)
    {
        /* Reset hooks */
        global_hooks.allocate = malloc;
        global_hooks.deallocate = free;
        global_hooks.reallocate = realloc;
        return;
    }

    global_hooks.allocate = malloc;
    if (hooks->malloc_fn != NULL)
    {
        global_hooks.allocate = hooks->malloc_fn;
    }

    global_hooks.deallocate = free;
    if (hooks->free_fn != NULL)
    {
        global_hooks.deallocate = hooks->free_fn;
    }

    /* use realloc only if both free and malloc are used */
    global_hooks.reallocate = NULL;
    if ((global_hooks.allocate == malloc) && (global_hooks.deallocate == free))
    {
        global_hooks.reallocate = realloc;
    }
}

/* Internal constructor. */
static cJSON *cJSON_New_Item(const internal_hooks * const hooks)
{
    cJSON* node = (cJSON*)hooks->allocate(sizeof(cJSON));
    if (node)
    {
        memset(node, '\0', sizeof(cJSON));
    }

    return node;
}

/* Delete a cJSON structure. */
cJSON_API void cJSON_Delete(cJSON *item)
{
    cJSON *next = NULL;
    while (item != NULL)
    {
        next = item->next;
        if (!(item->type & cJSON_IsReference) && (item->child != NULL))
        {
            cJSON_Delete(item->child);
        }
        if (!(item->type & cJSON_IsReference) && (item->valuestring != NULL))
        {
            global_hooks.deallocate(item->valuestring);
        }
        if (!(item->type & cJSON_StringIsConst) && (item->string != NULL))
        {
            global_hooks.deallocate(item->string);
        }
        global_hooks.deallocate(item);
        item = next;
    }
}

/* get the decimal point character of the current locale */
static unsigned char get_decimal_point(void)
{
#ifdef CJSON_LOCALE
    struct lconv *lconv = localeconv();
    return (unsigned char) lconv->decimal_point[0];
#else
    return '.';
#endif
}

typedef struct
{
    const unsigned char *content;
    size_t length;
    size_t offset;
    size_t depth; /* How deeply nested (in arrays/objects) is the parser? */
    internal_hooks hooks;
} parse_buffer;

/* check if the given size is left to read in a given parse buffer */
static cJSON_bool can_read(const parse_buffer * const buffer, size_t size)
{
    if (buffer == NULL)
    {
        return false;
    }

    return (buffer->offset + size) <= buffer->length;
}

/* get a pointer to the current position of the parser */
static const unsigned char *get_and_check_read_ptr(const parse_buffer * const buffer, size_t size)
{
    if (buffer == NULL)
    {
        return NULL;
    }

    if (!can_read(buffer, size)) {
        return NULL;
    }

    return buffer->content + buffer->offset;
}

/* get a pointer to the current position of the parser and advance the pointer by size bytes */
static const unsigned char *get_and_advance_read_ptr(parse_buffer * const buffer, size_t size)
{
    const unsigned char *read_ptr = NULL;

    if (buffer == NULL)
    {
        return NULL;
    }

    read_ptr = get_and_check_read_ptr(buffer, size);
    if (read_ptr != NULL)
    {
        buffer->offset += size;
    }

    return read_ptr;
}

/* Parse the input text to generate a number, and populate the result into item. */
static cJSON_bool parse_number(cJSON * const item, parse_buffer * const buffer)
{
    double number = 0;
    unsigned char *after_end = NULL;
    const unsigned char *number_ptr = NULL;
    unsigned char decimal_point = get_decimal_point();
    size_t i = 0;

    if (buffer == NULL)
    {
        return false;
    }

    number_ptr = get_and_check_read_ptr(buffer, 0);

    /* copy the number into a temporary buffer and replace '.' with the decimal point
     * of the current locale (for strtod) */
    for (i = 0; (i < (buffer->length - buffer->offset)) && (number_ptr[i] != '\0'); i++)
    {
        if ((number_ptr[i] >= '0') && (number_ptr[i] <= '9'))
        {
            continue;
        }
        if ((number_ptr[i] == '-') || (number_ptr[i] == '+') || (number_ptr[i] == 'e') || (number_ptr[i] == 'E'))
        {
            continue;
        }
        if (number_ptr[i] == decimal_point)
        {
            continue;
        }
        /* other characters are not valid numbers */
        goto fail;
    }

    number = strtod((const char*)number_ptr, (char**)&after_end);
    if (number_ptr == after_end)
    {
        return false; /* parse_error */
    }

    item->valuedouble = number;

    /* use saturation in case of overflow */
    if (number >= INT_MAX)
    {
        item->valueint = INT_MAX;
    }
    else if (number <= (double)INT_MIN)
    {
        item->valueint = INT_MIN;
    }
    else
    {
        item->valueint = (int)number;
    }

    item->type = cJSON_Number;

    buffer->offset += (size_t)(after_end - number_ptr);
    return true;

fail:
    return false;
}

/* don't ask me, but the original cJSON_SetNumberValue returns an integer or double. */
cJSON_API double cJSON_SetNumberHelper(cJSON *object, double number)
{
    if (object == NULL)
    {
        return number;
    }

    if (number >= INT_MAX)
    {
        object->valueint = INT_MAX;
    }
    else if (number <= (double)INT_MIN)
    {
        object->valueint = INT_MIN;
    }
    else
    {
        object->valueint = (int)number;
    }

    return object->valuedouble = number;
}

cJSON_API char* cJSON_SetValuestring(cJSON *object, const char *valuestring)
{
    char *copy = NULL;
    /* if object's type is not cJSON_String or is cJSON_IsReference, it should not set valuestring */
    if (!(object->type & cJSON_String) || (object->type & cJSON_IsReference))
    {
        return NULL;
    }
    if (strlen(valuestring) <= strlen(object->valuestring))
    {
        strcpy(object->valuestring, valuestring);
        return object->valuestring;
    }
    copy = (char*)cJSON_strdup((const unsigned char*)valuestring, &global_hooks);
    if (copy == NULL)
    {
        return NULL;
    }
    if (object->valuestring != NULL)
    {
        cJSON_free(object->valuestring);
    }
    object->valuestring = copy;

    return copy;
}

typedef struct
{
    unsigned char *buffer;
    size_t length;
    size_t offset;
    size_t depth; /* current nesting depth (for formatted printing) */
    cJSON_bool noalloc;
    cJSON_bool format; /* is this print a formatted print */
    internal_hooks hooks;
} printbuffer;

/* realloc printbuffer if necessary */
static unsigned char* ensure(printbuffer * const p, size_t needed)
{
    unsigned char *newbuffer = NULL;
    size_t newsize = 0;

    if (p == NULL)
    {
        return NULL;
    }

    if (needed > INT_MAX)
    {
        /* sizes bigger than INT_MAX are currently not supported */
        return NULL;
    }

    if (needed <= (p->length - p->offset))
    {
        return p->buffer + p->offset;
    }

    if (p->noalloc) {
        return NULL;
    }

    /* calculate new buffer size */
    newsize = p->length * 2;
    if (newsize < (p->offset + needed))
    {
        newsize = p->offset + needed;
    }

    if (p->hooks.reallocate != NULL)
    {
        /* reallocate with realloc if available */
        newbuffer = (unsigned char*)p->hooks.reallocate(p->buffer, newsize);
        if (newbuffer == NULL)
        {
            p->hooks.deallocate(p->buffer);
            p->length = 0;
            p->buffer = NULL;

            return NULL;
        }
    }
    else
    {
        /* otherwise reallocate manually */
        newbuffer = (unsigned char*)p->hooks.allocate(newsize);
        if (!newbuffer)
        {
            p->hooks.deallocate(p->buffer);
            p->length = 0;
            p->buffer = NULL;

            return NULL;
        }
        if (p->buffer != NULL)
        {
            memcpy(newbuffer, p->buffer, p->offset);
            p->hooks.deallocate(p->buffer);
        }
    }
    p->length = newsize;
    p->buffer = newbuffer;

    return newbuffer + p->offset;
}

/* calculate the new length of the string in a printbuffer and update the offset */
static void update_offset(printbuffer * const buffer)
{
    const unsigned char *buffer_pointer = NULL;
    if ((buffer == NULL) || (buffer->buffer == NULL))
    {
        return;
    }
    buffer_pointer = buffer->buffer + buffer->offset;

    buffer->offset += strlen((const char*)buffer_pointer);
}

/* securely comparison of floating-point variables */
static cJSON_bool compare_double(double a, double b)
{
    double maxVal = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return (fabs(a - b) <= maxVal * DBL_EPSILON);
}

/* Render the number nicely from the given item into a string. */
static cJSON_bool print_number(const cJSON * const item, printbuffer * const p)
{
    unsigned char *output_pointer = NULL;
    double d = item->valuedouble;
    int length = 0;
    size_t i = 0;
    unsigned char number_buffer[26] = {0}; /* temporary buffer to print the number into */
    unsigned char decimal_point = get_decimal_point();
    double test = 0.0;

    if (p == NULL)
    {
        return false;
    }

    /* This checks for NaN and Infinity */
    if (isnan(d) || isinf(d))
    {
        length = sprintf((char*)number_buffer, "null");
    }
    else
    {
        /* Try 15 decimal places of precision to avoid nonsensical garbage from strtod() */
        length = sprintf((char*)number_buffer, "%.15g", d);

        /* Check whether the original double can be recovered */
        if ((sscanf((char*)number_buffer, "%lg", &test) != 1) || !compare_double((double)test, d))
        {
            /* If not, print with 17 decimal places of precision */
            length = sprintf((char*)number_buffer, "%.17g", d);
        }
    }

    /* sprintf failed or buffer overrun occurred */
    if ((length < 0) || (length > (int)(sizeof(number_buffer) - 1)))
    {
        return false;
    }

    /* See if we need to replace a comma. */
    for (i = 0; i < ((size_t)length); i++)
    {
        if (number_buffer[i] == decimal_point)
        {
            number_buffer[i] = '.';
            break;
        }
    }

    output_pointer = ensure(p, (size_t)length);
    if (output_pointer == NULL)
    {
        return false;
    }

    memcpy(output_pointer, number_buffer, (size_t)length);
    p->offset += (size_t)length;

    return true;
}

/* Parse the input text into an unescaped cinput, and populate item. */
static const unsigned char *parse_string(cJSON * const item, parse_buffer * const buffer)
{
    const unsigned char *input_ptr = get_and_check_read_ptr(buffer, 1);
    const unsigned char *input_end = get_and_check_read_ptr(buffer, 1);
    unsigned char *output_ptr = NULL;
    unsigned char *output = NULL;

    /* not a string */
    if (*input_ptr != '\"')
    {
        goto fail;
    }

    {
        /* calculate approximate size of the output (overestimate) */
        size_t allocation_length = 0;
        size_t skipped_bytes = 0;
        input_end = input_ptr + 1;
        while (((size_t)(input_end - buffer->content) < buffer->length) && (*input_end != '\"'))
        {
            /* is it an escape sequence? */
            if (*input_end == '\\')
            {
                if ((size_t)(input_end + 1 - buffer->content) >= buffer->length)
                {
                    /* prevent buffer overflow when last input character is a backslash */
                    goto fail;
                }
                skipped_bytes++;
                input_end++;
            }
            input_end++;
        }
        if (((size_t)(input_end - buffer->content) >= buffer->length) || (*input_end != '\"'))
        {
            goto fail; /* string ended unexpectedly */
        }

        /* This is at most how much we need for the output */
        allocation_length = (size_t)(input_end - (input_ptr + 1)) - skipped_bytes;
        output = (unsigned char*)buffer->hooks.allocate(allocation_length + sizeof(""));
        if (output == NULL)
        {
            goto fail; /* allocation failure */
        }
    }

    output_ptr = output;
    /* loop through the string literal */
    while (input_ptr < input_end)
    {
        input_ptr++;
        if (*input_ptr == '\\')
        {
            unsigned char sequence_length = 2;
            switch (input_ptr[1])
            {
                case 'b':
                    *output_ptr++ = '\b';
                    break;
                case 'f':
                    *output_ptr++ = '\f';
                    break;
                case 'n':
                    *output_ptr++ = '\n';
                    break;
                case 'r':
                    *output_ptr++ = '\r';
                    break;
                case 't':
                    *output_ptr++ = '\t';
                    break;
                case '\"':
                case '\\':
                case '/':
                    *output_ptr++ = input_ptr[1];
                    break;

                /* UTF-16 literal */
                case 'u':
                    sequence_length = 5; /* \uXXXX */
                    if ((input_end - input_ptr) < (ptrdiff_t)sequence_length)
                    {
                        goto fail;
                    }
                    {
                        unsigned int codepoint = 0;
                        /* get the unicode codepoint */
                        if (sscanf((const char*)(input_ptr + 2), "%4x", &codepoint) != 1)
                        {
                            goto fail;
                        }

                        /* check if it is a surrogate pair */
                        if ((codepoint >= 0xD800) && (codepoint <= 0xDBFF))
                        {
                            unsigned int surrogate_codepoint = 0;
                            sequence_length = 11; /* \uXXXX\uYYYY */
                            if ((input_end - input_ptr) < (ptrdiff_t)sequence_length)
                            {
                                goto fail;
                            }
                            if ((input_ptr[6] != '\\') || (input_ptr[7] != 'u'))
                            {
                                goto fail;
                            }
                            /* get the second surrogate pair */
                            if (sscanf((const char*)(input_ptr + 8), "%4x", &surrogate_codepoint) != 1)
                            {
                                goto fail;
                            }
                            if ((surrogate_codepoint < 0xDC00) || (surrogate_codepoint > 0xDFFF))
                            {
                                goto fail;
                            }

                            codepoint = 0x10000 + (((codepoint & 0x3FF) << 10) | (surrogate_codepoint & 0x3FF));
                        }

                        /* encode as UTF-8 */
                        if (codepoint < 0x80)
                        {
                            /* normal ascii, encoding 0xxxxxxx */
                            *output_ptr++ = (unsigned char)codepoint;
                        }
                        else if (codepoint < 0x800)
                        {
                            /* two bytes, encoding 110xxxxx 10xxxxxx */
                            *output_ptr++ = (unsigned char)(0xC0 | (codepoint >> 6));
                            *output_ptr++ = (unsigned char)(0x80 | (codepoint & 0x3F));
                        }
                        else if (codepoint < 0x10000)
                        {
                            /* three bytes, encoding 1110xxxx 10xxxxxx 10xxxxxx */
                            *output_ptr++ = (unsigned char)(0xE0 | (codepoint >> 12));
                            *output_ptr++ = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
                            *output_ptr++ = (unsigned char)(0x80 | (codepoint & 0x3F));
                        }
                        else
                        {
                            /* four bytes, encoding 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx */
                            *output_ptr++ = (unsigned char)(0xF0 | (codepoint >> 18));
                            *output_ptr++ = (unsigned char)(0x80 | ((codepoint >> 12) & 0x3F));
                            *output_ptr++ = (unsigned char)(0x80 | ((codepoint >> 6) & 0x3F));
                            *output_ptr++ = (unsigned char)(0x80 | (codepoint & 0x3F));
                        }
                    }
                    break;

                default:
                    goto fail;
            }
            input_ptr += sequence_length - 1;
        }
        else
        {
            *output_ptr++ = *input_ptr;
        }
    }
    *output_ptr = '\0';

    item->type = cJSON_String;
    item->valuestring = (char*)output;

    buffer->offset = (size_t)(input_end - buffer->content);
    buffer->offset++;

    return buffer->content + buffer->offset;

fail:
    if (output != NULL)
    {
        buffer->hooks.deallocate(output);
    }

    if (input_ptr != NULL)
    {
        buffer->offset = (size_t)(input_ptr - buffer->content);
    }

    return NULL;
}

/* Render the cstring provided to an escaped version that can be printed. */
static cJSON_bool print_string_ptr(const unsigned char * const input, printbuffer * const p)
{
    const unsigned char *input_pointer = input;
    unsigned char *output = NULL;
    unsigned char *output_pointer = NULL;
    size_t output_length = 0;
    /* numbers of additional characters needed for escaping */
    size_t escape_characters = 0;

    if (p == NULL)
    {
        return false;
    }

    /* empty string */
    if (input == NULL)
    {
        output = ensure(p, static_strlen("\"\""));
        if (output == NULL)
        {
            return false;
        }
        strcpy((char*)output, "\"\"");

        return true;
    }

    /* set "flag" to 1 if something needs to be escaped */
    while (*input_pointer)
    {
        if ((*input_pointer > 31) && (*input_pointer != '\"') && (*input_pointer != '\\'))
        {
            input_pointer++;
        }
        else
        {
            /* these characters have to be escaped */
            escape_characters++;
            if ((*input_pointer == '\b') || (*input_pointer == '\f') || (*input_pointer == '\n') || (*input_pointer == '\r') || (*input_pointer == '\t'))
            {
                /* one character escape sequence */
                escape_characters++;
            }
            else
            {
                /* two character escape sequence */
                escape_characters += 2;
            }
            input_pointer++;
        }
    }
    output_length = (size_t)(input_pointer - input) + escape_characters;

    output = ensure(p, output_length + static_strlen("\"\""));
    if (output == NULL)
    {
        return false;
    }

    /* no characters have to be escaped */
    if (escape_characters == 0)
    {
        output[0] = '\"';
        memcpy(output + 1, input, output_length);
        output[output_length + 1] = '\"';
        output[output_length + 2] = '\0';

        return true;
    }

    output[0] = '\"';
    output_pointer = output + 1;
    /* copy the string */
    input_pointer = input;
    while (*input_pointer)
    {
        if ((*input_pointer > 31) && (*input_pointer != '\"') && (*input_pointer != '\\'))
        {
            *output_pointer++ = *input_pointer++;
        }
        else
        {
            *output_pointer++ = '\\';
            switch (*input_pointer++)
            {
                case '\\':
                    *output_pointer++ = '\\';
                    break;
                case '\"':
                    *output_pointer++ = '\"';
                    break;
                case '\b':
                    *output_pointer++ = 'b';
                    break;
                case '\f':
                    *output_pointer++ = 'f';
                    break;
                case '\n':
                    *output_pointer++ = 'n';
                    break;
                case '\r':
                    *output_pointer++ = 'r';
                    break;
                case '\t':
                    *output_pointer++ = 't';
                    break;
                default:
                    /* This should not happen, if it does, maybe enabling this will help debugging */
                    /*
                    sprintf((char*)output_pointer, "u%04x", *input_pointer);
                    output_pointer += 5;
                    */
                    break;
            }
        }
    }
    *output_pointer++ = '\"';
    *output_pointer = '\0';

    return true;
}

/* Invoke print_string_ptr (which is useful) on an item. */
static cJSON_bool print_string(const cJSON * const item, printbuffer * const p)
{
    return print_string_ptr((unsigned char*)item->valuestring, p);
}

/* Predeclare these prototypes. */
static cJSON_bool parse_value(cJSON * const item, parse_buffer * const buffer);
static cJSON_bool print_value(const cJSON * const item, printbuffer * const p);
static cJSON_bool parse_array(cJSON * const item, parse_buffer * const buffer);
static cJSON_bool print_array(const cJSON * const item, printbuffer * const p);
static cJSON_bool parse_object(cJSON * const item, parse_buffer * const buffer);
static cJSON_bool print_object(const cJSON * const item, printbuffer * const p);

/* Utility to jump whitespace and cr/lf */
static parse_buffer *buffer_skip_whitespace(parse_buffer * const buffer)
{
    const unsigned char *last_address = NULL;
    const unsigned char *current_address = NULL;

    if ((buffer == NULL) || (buffer->content == NULL))
    {
        return NULL;
    }

    last_address = buffer->content + buffer->length;
    current_address = buffer->content + buffer->offset;

    while ((current_address < last_address) && (*current_address <= 32))
    {
        current_address++;
    }

    if (current_address == last_address)
    {
        buffer->offset = buffer->length;
    }
    else
    {
        buffer->offset = (size_t)(current_address - buffer->content);
    }

    return buffer;
}

/* skip the UTF-8 BOM (byte order mark) if it is at the beginning of a buffer */
static parse_buffer *skip_utf8_bom(parse_buffer * const buffer)
{
    if ((buffer != NULL) && (buffer->content != NULL) && (buffer->offset == 0) && can_read(buffer, 3) && (strncmp((const char*)buffer->content, "\xEF\xBB\xBF", 3) == 0))
    {
        buffer->offset += 3;
    }
    return buffer;
}

cJSON_API cJSON *cJSON_ParseWithOpts(const char *value, const char **return_parse_end, cJSON_bool require_null_terminated)
{
    size_t buffer_length;

    if (NULL == value)
    {
        return NULL;
    }

    /* Adding null character size due to require_null_terminated. */
    buffer_length = strlen(value) + sizeof("");

    return cJSON_ParseWithLengthOpts(value, buffer_length, return_parse_end, require_null_terminated);
}

cJSON_API cJSON *cJSON_ParseWithLengthOpts(const char *value, size_t buffer_length, const char **return_parse_end, cJSON_bool require_null_terminated)
{
    parse_buffer buffer = { 0, 0, 0, 0, { 0, 0, 0 } };
    cJSON *item = NULL;

    /* reset error position */
    global_error.json = NULL;
    global_error.position = 0;

    if (value == NULL)
    {
        goto fail;
    }

    buffer.content = (const unsigned char*)value;
    buffer.length = buffer_length;
    buffer.offset = 0;
    buffer.hooks = global_hooks;

    item = cJSON_New_Item(&global_hooks);
    if (item == NULL) /* memory fail */
    {
        goto fail;
    }

    if (!parse_value(item, skip_utf8_bom(buffer_skip_whitespace(&buffer))))
    {
        /* parse failure. ep is set. */
        goto fail;
    }

    /* if we require null-terminated JSON without appended garbage, skip and then check for a null terminator */
    if (require_null_terminated)
    {
        buffer_skip_whitespace(&buffer);
        if ((buffer.offset < buffer.length) && (buffer.content[buffer.offset] != '\0'))
        {
            goto fail;
        }
    }
    if (return_parse_end)
    {
        *return_parse_end = (const char*)buffer.content + buffer.offset;
    }

    return item;

fail:
    if (item != NULL)
    {
        cJSON_Delete(item);
    }

    if (value != NULL)
    {
        error local_error;
        local_error.json = (const unsigned char*)value;
        local_error.position = 0;

        if (buffer.offset < buffer.length)
        {
            local_error.position = buffer.offset;
        }
        else if (buffer.length > 0)
        {
            local_error.position = buffer.length - 1;
        }

        if (return_parse_end != NULL)
        {
            *return_parse_end = (const char*)local_error.json + local_error.position;
        }

        global_error = local_error;
    }

    return NULL;
}

/* Default options for cJSON_Parse */
cJSON_API cJSON *cJSON_Parse(const char *value)
{
    return cJSON_ParseWithOpts(value, 0, 0);
}

cJSON_API cJSON *cJSON_ParseWithLength(const char *value, size_t buffer_length)
{
    return cJSON_ParseWithLengthOpts(value, buffer_length, 0, 0);
}

#define cjson_min(a, b) (((a) < (b)) ? (a) : (b))

static unsigned char *print(const cJSON * const item, cJSON_bool format, const internal_hooks * const hooks)
{
    static const size_t default_buffer_size = 256;
    printbuffer p = { 0, 0, 0, 0, 0, 0, { 0, 0, 0 } };
    unsigned char *printed = NULL;

    /* create buffer */
    p.buffer = (unsigned char*)hooks->allocate(default_buffer_size);
    p.length = default_buffer_size;
    p.format = format;
    p.hooks = *hooks;
    if (p.buffer == NULL)
    {
        goto fail;
    }

    /* print the value */
    if (!print_value(item, &p))
    {
        goto fail;
    }
    update_offset(&p);

    /* check if reallocate is available */
    if (hooks->reallocate != NULL)
    {
        printed = (unsigned char*)hooks->reallocate(p.buffer, p.offset + 1);
        if (printed == NULL) {
            goto fail;
        }
        p.buffer = NULL;
    }
    else /* otherwise copy the buffer */
    {
        printed = (unsigned char*)hooks->allocate(p.offset + 1);
        if (printed == NULL)
        {
            goto fail;
        }
        memcpy(printed, p.buffer, cjson_min(p.length, p.offset + 1));
        printed[p.offset] = '\0'; /* just to be sure */

        /* free the buffer */
        hooks->deallocate(p.buffer);
    }

    return printed;

fail:
    if (p.buffer != NULL)
    {
        hooks->deallocate(p.buffer);
    }

    if (printed != NULL)
    {
        hooks->deallocate(printed);
    }

    return NULL;
}

/* Render a cJSON item to text */
cJSON_API char *cJSON_Print(const cJSON *item)
{
    return (char*)print(item, true, &global_hooks);
}

cJSON_API char *cJSON_PrintUnformatted(const cJSON *item)
{
    return (char*)print(item, false, &global_hooks);
}

cJSON_API char *cJSON_PrintBuffered(const cJSON *item, int prebuffer, cJSON_bool fmt)
{
    printbuffer p = { 0, 0, 0, 0, 0, 0, { 0, 0, 0 } };

    if (prebuffer < 0)
    {
        return NULL;
    }

    p.buffer = (unsigned char*)global_hooks.allocate((size_t)prebuffer);
    if (!p.buffer)
    {
        return NULL;
    }

    p.length = (size_t)prebuffer;
    p.offset = 0;
    p.noalloc = false;
    p.format = fmt;
    p.hooks = global_hooks;

    if (!print_value(item, &p))
    {
        global_hooks.deallocate(p.buffer);
        return NULL;
    }

    return (char*)p.buffer;
}

cJSON_API cJSON_bool cJSON_PrintPreallocated(cJSON *item, char *buffer, const int length, const cJSON_bool format)
{
    printbuffer p = { 0, 0, 0, 0, 0, 0, { 0, 0, 0 } };

    if ((length < 0) || (buffer == NULL))
    {
        return false;
    }

    p.buffer = (unsigned char*)buffer;
    p.length = (size_t)length;
    p.offset = 0;
    p.noalloc = true;
    p.format = format;
    p.hooks = global_hooks;

    if (!print_value(item, &p))
    {
        return false;
    }

    /* ensure null termination */
    if (p.offset >= p.length)
    {
        return false;
    }
    p.buffer[p.offset] = '\0';

    return true;
}

/* Parser core - when encountering text, process appropriately. */
static cJSON_bool parse_value(cJSON * const item, parse_buffer * const buffer)
{
    if ((buffer == NULL) || (buffer->content == NULL))
    {
        return false; /* no input */
    }

    /* parse the different types of values */
    /* null */
    if (can_read(buffer, 4) && (strncmp((const char*)get_and_check_read_ptr(buffer, 0), "null", 4) == 0))
    {
        item->type = cJSON_NULL;
        buffer->offset += 4;
        return true;
    }
    /* false */
    if (can_read(buffer, 5) && (strncmp((const char*)get_and_check_read_ptr(buffer, 0), "false", 5) == 0))
    {
        item->type = cJSON_False;
        buffer->offset += 5;
        return true;
    }
    /* true */
    if (can_read(buffer, 4) && (strncmp((const char*)get_and_check_read_ptr(buffer, 0), "true", 4) == 0))
    {
        item->type = cJSON_True;
        item->valueint = 1;
        buffer->offset += 4;
        return true;
    }
    /* string */
    if (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == '\"'))
    {
        return parse_string(item, buffer) != NULL;
    }
    /* number */
    if (can_read(buffer, 1) && ((*get_and_check_read_ptr(buffer, 0) == '-') || ((*get_and_check_read_ptr(buffer, 0) >= '0') && (*get_and_check_read_ptr(buffer, 0) <= '9'))))
    {
        return parse_number(item, buffer);
    }
    /* array */
    if (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == '['))
    {
        return parse_array(item, buffer);
    }
    /* object */
    if (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == '{'))
    {
        return parse_object(item, buffer);
    }

    return false;
}

/* Render a value to text. */
static cJSON_bool print_value(const cJSON * const item, printbuffer * const p)
{
    unsigned char *output = NULL;

    if ((item == NULL) || (p == NULL))
    {
        return false;
    }

    switch ((item->type) & 0xFF)
    {
        case cJSON_NULL:
            output = ensure(p, static_strlen("null"));
            if (output == NULL)
            {
                return false;
            }
            strcpy((char*)output, "null");
            return true;

        case cJSON_False:
            output = ensure(p, static_strlen("false"));
            if (output == NULL)
            {
                return false;
            }
            strcpy((char*)output, "false");
            return true;

        case cJSON_True:
            output = ensure(p, static_strlen("true"));
            if (output == NULL)
            {
                return false;
            }
            strcpy((char*)output, "true");
            return true;

        case cJSON_Number:
            return print_number(item, p);

        case cJSON_Raw:
        {
            size_t raw_length = 0;
            if (item->valuestring == NULL)
            {
                return false;
            }

            raw_length = strlen(item->valuestring) + sizeof("");
            output = ensure(p, raw_length);
            if (output == NULL)
            {
                return false;
            }
            memcpy(output, item->valuestring, raw_length);
            return true;
        }

        case cJSON_String:
            return print_string(item, p);

        case cJSON_Array:
            return print_array(item, p);

        case cJSON_Object:
            return print_object(item, p);

        default:
            return false;
    }
}

/* Build an array from input text. */
static cJSON_bool parse_array(cJSON * const item, parse_buffer * const buffer)
{
    cJSON *head = NULL; /* head of the linked list */
    cJSON *current_item = NULL;

    if (buffer->depth >= CJSON_NESTING_LIMIT)
    {
        return false; /* reached nesting limit */
    }
    buffer->depth++;

    if (*get_and_advance_read_ptr(buffer, 1) != '[')
    {
        /* not an array */
        goto fail;
    }
    buffer_skip_whitespace(buffer);

    if (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == ']'))
    {
        /* empty array */
        goto success;
    }

    /* check if we skipped to the end of the buffer */
    if (!can_read(buffer, 1))
    {
        buffer->offset--;
        goto fail;
    }

    /* step back to character in front of the first element */
    buffer->offset--;
    /* loop through the comma separated array elements */
    do
    {
        /* allocate next item */
        cJSON *new_item = cJSON_New_Item(&(buffer->hooks));
        if (new_item == NULL)
        {
            goto fail; /* memory fail */
        }

        /* attach next item to list */
        if (head == NULL)
        {
            /* start the linked list */
            current_item = head = new_item;
        }
        else
        {
            /* add to the end and advance */
            current_item->next = new_item;
            new_item->prev = current_item;
            current_item = new_item;
        }

        /* parse next value */
        buffer->offset++;
        buffer_skip_whitespace(buffer);
        if (!parse_value(current_item, buffer))
        {
            goto fail; /* failed to parse value */
        }
        buffer_skip_whitespace(buffer);
    }
    while (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == ','));

    if (!can_read(buffer, 1) || (*get_and_check_read_ptr(buffer, 0) != ']'))
    {
        goto fail; /* expected end of array */
    }

success:
    buffer->depth--;

    if (head != NULL) {
        head->prev = current_item;
    }

    item->type = cJSON_Array;
    item->child = head;

    buffer->offset++;

    return true;

fail:
    if (head != NULL)
    {
        cJSON_Delete(head);
    }

    return false;
}

/* Render an array to text */
static cJSON_bool print_array(const cJSON * const item, printbuffer * const p)
{
    unsigned char *output_pointer = NULL;
    size_t length = 0;
    cJSON *current_element = item->child;

    if (p == NULL)
    {
        return false;
    }

    /* Compose the output array. */
    /* opening square bracket */
    output_pointer = ensure(p, 1);
    if (output_pointer == NULL)
    {
        return false;
    }

    *output_pointer = '[';
    p->offset++;
    p->depth++;

    while (current_element != NULL)
    {
        if (!print_value(current_element, p))
        {
            return false;
        }
        update_offset(p);
        if (current_element->next)
        {
            length = p->format ? 2 : 1;
            output_pointer = ensure(p, length + 1);
            if (output_pointer == NULL)
            {
                return false;
            }
            *output_pointer++ = ',';
            if(p->format)
            {
                *output_pointer++ = ' ';
            }
            *output_pointer = '\0';
            p->offset += length;
        }
        current_element = current_element->next;
    }

    output_pointer = ensure(p, 2);
    if (output_pointer == NULL)
    {
        return false;
    }
    if (p->format)
    {
        *output_pointer++ = '\n';
        p->depth--;
        for (length = 0; length < p->depth; length++)
        {
            *output_pointer++ = '\t';
        }
    }
    *output_pointer++ = ']';
    *output_pointer = '\0';

    return true;
}

/* Build an object from the text. */
static cJSON_bool parse_object(cJSON * const item, parse_buffer * const buffer)
{
    cJSON *head = NULL; /* head of the linked list */
    cJSON *current_item = NULL;

    if (buffer->depth >= CJSON_NESTING_LIMIT)
    {
        return false; /* reached nesting limit */
    }
    buffer->depth++;

    if (!can_read(buffer, 1) || (*get_and_advance_read_ptr(buffer, 1) != '{'))
    {
        goto fail; /* not an object */
    }
    buffer_skip_whitespace(buffer);

    if (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == '}'))
    {
        goto success; /* empty object */
    }

    /* check if we skipped to the end of the buffer */
    if (!can_read(buffer, 1))
    {
        buffer->offset--;
        goto fail;
    }

    /* step back to character in front of the first element */
    buffer->offset--;
    /* loop through the comma separated array elements */
    do
    {
        /* allocate next item */
        cJSON *new_item = cJSON_New_Item(&(buffer->hooks));
        if (new_item == NULL)
        {
            goto fail; /* memory fail */
        }

        /* attach next item to list */
        if (head == NULL)
        {
            /* start the linked list */
            current_item = head = new_item;
        }
        else
        {
            /* add to the end and advance */
            current_item->next = new_item;
            new_item->prev = current_item;
            current_item = new_item;
        }

        /* parse the name of the child */
        buffer->offset++;
        buffer_skip_whitespace(buffer);
        if (parse_string(current_item, buffer) == NULL)
        {
            goto fail; /* failed to parse name */
        }
        buffer_skip_whitespace(buffer);

        /* swap valuestring and string, because we parsed the name */
        current_item->string = current_item->valuestring;
        current_item->valuestring = NULL;

        if (!can_read(buffer, 1) || (*get_and_advance_read_ptr(buffer, 1) != ':'))
        {
            goto fail; /* invalid object */
        }

        /* parse value */
        buffer_skip_whitespace(buffer);
        if (!parse_value(current_item, buffer))
        {
            goto fail; /* failed to parse value */
        }
        buffer_skip_whitespace(buffer);
    }
    while (can_read(buffer, 1) && (*get_and_check_read_ptr(buffer, 0) == ','));

    if (!can_read(buffer, 1) || (*get_and_check_read_ptr(buffer, 0) != '}'))
    {
        goto fail; /* expected end of object */
    }

success:
    buffer->depth--;

    if (head != NULL) {
        head->prev = current_item;
    }

    item->type = cJSON_Object;
    item->child = head;

    buffer->offset++;
    return true;

fail:
    if (head != NULL)
    {
        cJSON_Delete(head);
    }

    return false;
}

/* Render an object to text. */
static cJSON_bool print_object(const cJSON * const item, printbuffer * const p)
{
    unsigned char *output_pointer = NULL;
    size_t length = 0;
    cJSON *current_item = item->child;

    if (p == NULL)
    {
        return false;
    }

    /* Compose the output: */
    length = (p->format ? 2 : 1); /* {\n */
    output_pointer = ensure(p, length);
    if (output_pointer == NULL)
    {
        return false;
    }

    *output_pointer++ = '{';
    p->depth++;
    if (p->format)
    {
        *output_pointer++ = '\n';
    }
    p->offset += length;

    while (current_item)
    {
        if (p->format)
        {
            size_t i;
            output_pointer = ensure(p, p->depth);
            if (output_pointer == NULL)
            {
                return false;
            }
            for (i = 0; i < p->depth; i++)
            {
                *output_pointer++ = '\t';
            }
            p->offset += p->depth;
        }

        /* print key */
        if (!print_string_ptr((unsigned char*)current_item->string, p))
        {
            return false;
        }
        update_offset(p);

        length = p->format ? 2 : 1;
        output_pointer = ensure(p, length);
        if (output_pointer == NULL)
        {
            return false;
        }
        *output_pointer++ = ':';
        if (p->format)
        {
            *output_pointer++ = '\t';
        }
        p->offset += length;

        /* print value */
        if (!print_value(current_item, p))
        {
            return false;
        }
        update_offset(p);

        /* print comma if not last */
        length = ((size_t)(p->format ? 1 : 0) + 1);
        if (current_item->next)
        {
            output_pointer = ensure(p, length);
            if (output_pointer == NULL)
            {
                return false;
            }
            *output_pointer++ = ',';
            if (p->format)
            {
                *output_pointer++ = '\n';
            }
            *output_pointer = '\0';
            p->offset += length;
        }

        current_item = current_item->next;
    }

    output_pointer = ensure(p, p->format ? (p->depth + 1) : 2);
    if (output_pointer == NULL)
    {
        return false;
    }
    if (p->format)
    {
        p->depth--;
        for (length = 0; length < p->depth; length++)
        {
            *output_pointer++ = '\t';
        }
        *output_pointer++ = '\n';
    }
    *output_pointer++ = '}';
    *output_pointer = '\0';

    return true;
}

/* Get Array size/item / object item. */
cJSON_API int cJSON_GetArraySize(const cJSON *array)
{
    cJSON *child = NULL;
    int i = 0;

    if (array == NULL)
    {
        return 0;
    }

    child = array->child;

    while(child != NULL)
    {
        i++;
        child = child->next;
    }

    return i;
}

static cJSON* get_array_item(const cJSON *array, size_t index)
{
    cJSON *current_child = NULL;

    if (array == NULL)
    {
        return NULL;
    }

    current_child = array->child;
    while ((current_child != NULL) && (index > 0))
    {
        index--;
        current_child = current_child->next;
    }

    return current_child;
}

cJSON_API cJSON *cJSON_GetArrayItem(const cJSON *array, int index)
{
    if (index < 0)
    {
        return NULL;
    }

    return get_array_item(array, (size_t)index);
}

static cJSON *get_object_item(const cJSON * const object, const char * const name, const cJSON_bool case_sensitive)
{
    cJSON *current_element = NULL;

    if ((object == NULL) || (name == NULL))
    {
        return NULL;
    }

    current_element = object->child;
    if (case_sensitive)
    {
        while ((current_element != NULL) && (current_element->string != NULL) && (strcmp(name, current_element->string) != 0))
        {
            current_element = current_element->next;
        }
    }
    else
    {
        while ((current_element != NULL) && (case_insensitive_strcmp((const unsigned char*)name, (const unsigned char*)(current_element->string)) != 0))
        {
            current_element = current_element->next;
        }
    }

    return current_element;
}

cJSON_API cJSON *cJSON_GetObjectItem(const cJSON * const object, const char * const string)
{
    return get_object_item(object, string, false);
}

cJSON_API cJSON *cJSON_GetObjectItemCaseSensitive(const cJSON * const object, const char * const string)
{
    return get_object_item(object, string, true);
}

cJSON_API cJSON_bool cJSON_HasObjectItem(const cJSON *object, const char *string)
{
    return cJSON_GetObjectItem(object, string) ? 1 : 0;
}

/* Utility for array list handling. */
static void suffix_object(cJSON *prev, cJSON *item)
{
    prev->next = item;
    item->prev = prev;
}

/* Utility for handling references. */
static cJSON *create_reference(const cJSON *item, const internal_hooks * const hooks)
{
    cJSON *reference = NULL;
    if (item == NULL)
    {
        return NULL;
    }

    reference = cJSON_New_Item(hooks);
    if (reference == NULL)
    {
        return NULL;
    }

    memcpy(reference, item, sizeof(cJSON));
    reference->string = NULL;
    reference->type |= cJSON_IsReference;
    reference->next = reference->prev = NULL;
    return reference;
}

static cJSON_bool add_item_to_array(cJSON *array, cJSON *item)
{
    cJSON *child = NULL;

    if ((item == NULL) || (array == NULL) || (array->type != cJSON_Array))
    {
        return false;
    }

    child = array->child;

    if (child == NULL)
    {
        /* list is empty, start new one */
        array->child = item;
        item->prev = item;
        item->next = NULL;
    }
    else
    {
        /* append to the end */
        if (child->prev != NULL)
        {
            suffix_object(child->prev, item);
            array->child->prev = item;
        }
    }

    return true;
}

/* Add item to array/object. */
cJSON_API void cJSON_AddItemToArray(cJSON *array, cJSON *item)
{
    add_item_to_array(array, item);
}

#if defined(__clang__) || (defined(__GNUC__) && ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 5))))
    #pragma GCC diagnostic push
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wcast-qual"
#endif
/* Add an item to an object */
static cJSON_bool add_item_to_object(cJSON * const object, const char * const string, cJSON * const item, const internal_hooks * const hooks, const cJSON_bool constant_key)
{
    char *new_key = NULL;
    int new_type = cJSON_Invalid;

    if ((object == NULL) || (string == NULL) || (item == NULL) || (object->type != cJSON_Object))
    {
        return false;
    }

    if (constant_key)
    {
        new_key = (char*)string;
        new_type = cJSON_StringIsConst;
    }
    else
    {
        new_key = (char*)cJSON_strdup((const unsigned char*)string, hooks);
        if (new_key == NULL)
        {
            return false;
        }

        new_type = 0;
    }

    if (!(item->type & cJSON_StringIsConst) && (item->string != NULL))
    {
        hooks->deallocate(item->string);
    }

    item->string = new_key;
    item->type |= new_type;

    return add_item_to_array(object, item);
}
#if defined(__clang__) || (defined(__GNUC__) && ((__GNUC__ > 4) || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 5))))
    #pragma GCC diagnostic pop
#endif

cJSON_API void cJSON_AddItemToObject(cJSON *object, const char *string, cJSON *item)
{
    add_item_to_object(object, string, item, &global_hooks, false);
}

/* Add an item to an object with constant string as key */
cJSON_API void cJSON_AddItemToObjectCS(cJSON *object, const char *string, cJSON *item)
{
    add_item_to_object(object, string, item, &global_hooks, true);
}

cJSON_API void cJSON_AddItemReferenceToArray(cJSON *array, cJSON *item)
{
    if (array == NULL)
    {
        return;
    }

    add_item_to_array(array, create_reference(item, &global_hooks));
}

cJSON_API void cJSON_AddItemReferenceToObject(cJSON *object, const char *string, cJSON *item)
{
    if ((object == NULL) || (string == NULL))
    {
        return;
    }

    add_item_to_object(object, string, create_reference(item, &global_hooks), &global_hooks, false);
}

cJSON_API cJSON* cJSON_AddNullToObject(cJSON * const object, const char * const name)
{
    cJSON *null_item = cJSON_CreateNull();
    if (add_item_to_object(object, name, null_item, &global_hooks, false))
    {
        return null_item;
    }

    cJSON_Delete(null_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddTrueToObject(cJSON * const object, const char * const name)
{
    cJSON *true_item = cJSON_CreateTrue();
    if (add_item_to_object(object, name, true_item, &global_hooks, false))
    {
        return true_item;
    }

    cJSON_Delete(true_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddFalseToObject(cJSON * const object, const char * const name)
{
    cJSON *false_item = cJSON_CreateFalse();
    if (add_item_to_object(object, name, false_item, &global_hooks, false))
    {
        return false_item;
    }

    cJSON_Delete(false_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddBoolToObject(cJSON * const object, const char * const name, const cJSON_bool boolean)
{
    cJSON *bool_item = cJSON_CreateBool(boolean);
    if (add_item_to_object(object, name, bool_item, &global_hooks, false))
    {
        return bool_item;
    }

    cJSON_Delete(bool_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddNumberToObject(cJSON * const object, const char * const name, const double number)
{
    cJSON *number_item = cJSON_CreateNumber(number);
    if (add_item_to_object(object, name, number_item, &global_hooks, false))
    {
        return number_item;
    }

    cJSON_Delete(number_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddStringToObject(cJSON * const object, const char * const name, const char * const string)
{
    cJSON *string_item = cJSON_CreateString(string);
    if (add_item_to_object(object, name, string_item, &global_hooks, false))
    {
        return string_item;
    }

    cJSON_Delete(string_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddRawToObject(cJSON * const object, const char * const name, const char * const raw)
{
    cJSON *raw_item = cJSON_CreateRaw(raw);
    if (add_item_to_object(object, name, raw_item, &global_hooks, false))
    {
        return raw_item;
    }

    cJSON_Delete(raw_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddObjectToObject(cJSON * const object, const char * const name)
{
    cJSON *object_item = cJSON_CreateObject();
    if (add_item_to_object(object, name, object_item, &global_hooks, false))
    {
        return object_item;
    }

    cJSON_Delete(object_item);
    return NULL;
}

cJSON_API cJSON* cJSON_AddArrayToObject(cJSON * const object, const char * const name)
{
    cJSON *array_item = cJSON_CreateArray();
    if (add_item_to_object(object, name, array_item, &global_hooks, false))
    {
        return array_item;
    }

    cJSON_Delete(array_item);
    return NULL;
}

cJSON_API cJSON *cJSON_DetachItemViaPointer(cJSON *parent, cJSON * const item)
{
    if ((parent == NULL) || (item == NULL))
    {
        return NULL;
    }

    if (item->prev != NULL)
    {
        /* not the first element */
        item->prev->next = item->next;
    }
    if (item->next != NULL)
    {
        /* not the last element */
        item->next->prev = item->prev;
    }

    if (item == parent->child)
    {
        /* first element */
        parent->child = item->next;
    }
    else if (item->prev == parent->child)
    {
        /* last element */
        parent->child->prev = item->prev;
    }


    item->prev = NULL;
    item->next = NULL;

    return item;
}

cJSON_API cJSON *cJSON_DetachItemFromArray(cJSON *array, int which)
{
    if (which < 0)
    {
        return NULL;
    }

    return cJSON_DetachItemViaPointer(array, get_array_item(array, (size_t)which));
}

cJSON_API void cJSON_DeleteItemFromArray(cJSON *array, int which)
{
    cJSON_Delete(cJSON_DetachItemFromArray(array, which));
}

cJSON_API cJSON *cJSON_DetachItemFromObject(cJSON *object, const char *string)
{
    cJSON *to_detach = cJSON_GetObjectItem(object, string);

    return cJSON_DetachItemViaPointer(object, to_detach);
}

cJSON_API cJSON *cJSON_DetachItemFromObjectCaseSensitive(cJSON *object, const char *string)
{
    cJSON *to_detach = cJSON_GetObjectItemCaseSensitive(object, string);

    return cJSON_DetachItemViaPointer(object, to_detach);
}

cJSON_API void cJSON_DeleteItemFromObject(cJSON *object, const char *string)
{
    cJSON_Delete(cJSON_DetachItemFromObject(object, string));
}

cJSON_API void cJSON_DeleteItemFromObjectCaseSensitive(cJSON *object, const char *string)
{
    cJSON_Delete(cJSON_DetachItemFromObjectCaseSensitive(object, string));
}

/* Replace array/object items with new ones. */
cJSON_API void cJSON_InsertItemInArray(cJSON *array, int which, cJSON *newitem)
{
    cJSON *after_inserted = NULL;

    if (which < 0)
    {
        return;
    }

    after_inserted = get_array_item(array, (size_t)which);
    if (after_inserted == NULL)
    {
        add_item_to_array(array, newitem);
        return;
    }

    newitem->next = after_inserted;
    newitem->prev = after_inserted->prev;
    after_inserted->prev = newitem;
    if (after_inserted == array->child)
    {
        array->child = newitem;
    }
    else
    {
        newitem->prev->next = newitem;
    }
}

cJSON_API cJSON_bool cJSON_ReplaceItemViaPointer(cJSON * const parent, cJSON * const item, cJSON * replacement)
{
    if ((parent == NULL) || (replacement == NULL) || (item == NULL))
    {
        return false;
    }

    if (replacement == item)
    {
        return true;
    }

    replacement->next = item->next;
    replacement->prev = item->prev;

    if (replacement->next != NULL)
    {
        replacement->next->prev = replacement;
    }
    if (parent->child == item)
    {
        if (parent->child->prev == parent->child)
        {
            replacement->prev = replacement;
        }
        parent->child = replacement;
    }
    else
    {   /*
        // To find the last item in array quickly
        if (item->next == NULL)
        {
            parent->child->prev = replacement;
        }
        */
        if (replacement->prev != NULL)
        {
            replacement->prev->next = replacement;
        }
    }

    item->next = NULL;
    item->prev = NULL;
    cJSON_Delete(item);

    return true;
}

cJSON_API void cJSON_ReplaceItemInArray(cJSON *array, int which, cJSON *newitem)
{
    if (which < 0)
    {
        return;
    }

    cJSON_ReplaceItemViaPointer(array, get_array_item(array, (size_t)which), newitem);
}

static cJSON_bool replace_item_in_object(cJSON *object, const char *string, cJSON *replacement, cJSON_bool case_sensitive)
{
    if ((replacement == NULL) || (string == NULL))
    {
        return false;
    }

    /* replace the name in the replacement */
    if (!(replacement->type & cJSON_StringIsConst) && (replacement->string != NULL))
    {
        cJSON_free(replacement->string);
    }
    replacement->string = (char*)cJSON_strdup((const unsigned char*)string, &global_hooks);
    if (replacement->string == NULL)
    {
        return false;
    }

    cJSON_ReplaceItemViaPointer(object, get_object_item(object, string, case_sensitive), replacement);

    return true;
}

cJSON_API void cJSON_ReplaceItemInObject(cJSON *object, const char *string, cJSON *newitem)
{
    replace_item_in_object(object, string, newitem, false);
}

cJSON_API void cJSON_ReplaceItemInObjectCaseSensitive(cJSON *object, const char *string, cJSON *newitem)
{
    replace_item_in_object(object, string, newitem, true);
}

/* Create basic types: */
cJSON_API cJSON *cJSON_CreateNull(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_NULL;
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateTrue(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_True;
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateFalse(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_False;
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateBool(cJSON_bool boolean)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = boolean ? cJSON_True : cJSON_False;
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateNumber(double num)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_Number;
        item->valuedouble = num;

        /* use saturation in case of overflow */
        if (num >= INT_MAX)
        {
            item->valueint = INT_MAX;
        }
        else if (num <= (double)INT_MIN)
        {
            item->valueint = INT_MIN;
        }
        else
        {
            item->valueint = (int)num;
        }
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateString(const char *string)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_String;
        item->valuestring = (char*)cJSON_strdup((const unsigned char*)string, &global_hooks);
        if(!item->valuestring)
        {
            cJSON_Delete(item);
            return NULL;
        }
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateRaw(const char *raw)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type = cJSON_Raw;
        item->valuestring = (char*)cJSON_strdup((const unsigned char*)raw, &global_hooks);
        if(!item->valuestring)
        {
            cJSON_Delete(item);
            return NULL;
        }
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateArray(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if(item)
    {
        item->type=cJSON_Array;
    }

    return item;
}

cJSON_API cJSON *cJSON_CreateObject(void)
{
    cJSON *item = cJSON_New_Item(&global_hooks);
    if (item)
    {
        item->type = cJSON_Object;
    }

    return item;
}

/* Create Arrays: */
cJSON_API cJSON *cJSON_CreateIntArray(const int *numbers, int count)
{
    size_t i = 0;
    cJSON *n = NULL;
    cJSON *p = NULL;
    cJSON *a = NULL;

    if ((count < 0) || (numbers == NULL))
    {
        return NULL;
    }

    a = cJSON_CreateArray();

    for(i = 0; a && (i < (size_t)count); i++)
    {
        n = cJSON_CreateNumber(numbers[i]);
        if (!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if(!i)
        {
            a->child = n;
        }
        else
        {
            suffix_object(p, n);
        }
        p = n;
    }

    if (a && a->child)
    {
        a->child->prev = p;
    }


    return a;
}

cJSON_API cJSON *cJSON_CreateFloatArray(const float *numbers, int count)
{
    size_t i = 0;
    cJSON *n = NULL;
    cJSON *p = NULL;
    cJSON *a = NULL;

    if ((count < 0) || (numbers == NULL))
    {
        return NULL;
    }

    a = cJSON_CreateArray();

    for(i = 0; a && (i < (size_t)count); i++)
    {
        n = cJSON_CreateNumber((double)numbers[i]);
        if(!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if(!i)
        {
            a->child = n;
        }
        else
        {
            suffix_object(p, n);
        }
        p = n;
    }

    if (a && a->child)
    {
        a->child->prev = p;
    }

    return a;
}

cJSON_API cJSON *cJSON_CreateDoubleArray(const double *numbers, int count)
{
    size_t i = 0;
    cJSON *n = NULL;
    cJSON *p = NULL;
    cJSON *a = NULL;

    if ((count < 0) || (numbers == NULL))
    {
        return NULL;
    }

    a = cJSON_CreateArray();

    for(i = 0; a && (i < (size_t)count); i++)
    {
        n = cJSON_CreateNumber(numbers[i]);
        if(!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if(!i)
        {
            a->child = n;
        }
        else
        {
            suffix_object(p, n);
        }
        p = n;
    }

    if (a && a->child)
    {
        a->child->prev = p;
    }

    return a;
}

cJSON_API cJSON *cJSON_CreateStringArray(const char *const *strings, int count)
{
    size_t i = 0;
    cJSON *n = NULL;
    cJSON *p = NULL;
    cJSON *a = NULL;

    if ((count < 0) || (strings == NULL))
    {
        return NULL;
    }

    a = cJSON_CreateArray();

    for (i = 0; a && (i < (size_t)count); i++)
    {
        n = cJSON_CreateString(strings[i]);
        if(!n)
        {
            cJSON_Delete(a);
            return NULL;
        }
        if(!i)
        {
            a->child = n;
        }
        else
        {
            suffix_object(p,n);
        }
        p = n;
    }

    if (a && a->child)
    {
        a->child->prev = p;
    }

    return a;
}

cJSON_API cJSON *cJSON_Duplicate(const cJSON *item, cJSON_bool recurse)
{
    cJSON *newitem = NULL;
    cJSON *child = NULL;
    cJSON *next = NULL;
    cJSON *newchild = NULL;

    /* Bail out if item is NULL */
    if (!item)
    {
        goto fail;
    }
    /* Create new item */
    newitem = cJSON_New_Item(&global_hooks);
    if (!newitem)
    {
        goto fail;
    }
    /* Copy over all vars */
    newitem->type = item->type & (~cJSON_IsReference);
    newitem->valueint = item->valueint;
    newitem->valuedouble = item->valuedouble;
    if (item->valuestring)
    {
        newitem->valuestring = (char*)cJSON_strdup((unsigned char*)item->valuestring, &global_hooks);
        if (!newitem->valuestring)
        {
            goto fail;
        }
    }
    if (item->string)
    {
        newitem->string = (item->type&cJSON_StringIsConst) ? item->string : (char*)cJSON_strdup((unsigned char*)item->string, &global_hooks);
        if (!newitem->string)
        {
            goto fail;
        }
    }
    /* If recurse is non-zero, walk the sub-items. */
    if (recurse)
    {
        child = item->child;
        while (child != NULL)
        {
            newchild = cJSON_Duplicate(child, true); /* Duplicate (with recurse) each item in the appropriate structure */
            if (!newchild)
            {
                goto fail;
            }
            if (next != NULL)
            {
                /* If newitem is already exist just add it */
                next->next = newchild;
                newchild->prev = next;
                next = newchild;
            }
            else
            {
                /* Otherwise, start a new chain */
                newitem->child = newchild;
                next = newchild;
            }
            child = child->next;
        }
        if (newitem && newitem->child)
        {
            newitem->child->prev = newchild;
        }
    }

    return newitem;

fail:
    if (newitem != NULL)
    {
        cJSON_Delete(newitem);
    }

    return NULL;
}

cJSON_API cJSON_bool cJSON_Compare(const cJSON * const a, const cJSON * const b, const cJSON_bool case_sensitive)
{
    if ((a == NULL) || (b == NULL) || ((a->type & 0xFF) != (b->type & 0xFF)))
    {
        return false;
    }

    /* check if type is valid */
    switch (a->type & 0xFF)
    {
        case cJSON_False:
        case cJSON_True:
        case cJSON_NULL:
        case cJSON_Number:
        case cJSON_String:
        case cJSON_Raw:
        case cJSON_Array:
        case cJSON_Object:
            break;

        default:
            return false;
    }

    /* identical objects are equal */
    if (a == b)
    {
        return true;
    }

    switch (a->type & 0xFF)
    {
        case cJSON_False:
        case cJSON_True:
        case cJSON_NULL:
            return true;

        case cJSON_Number:
            if (compare_double(a->valuedouble, b->valuedouble))
            {
                return true;
            }
            return false;

        case cJSON_String:
        case cJSON_Raw:
            if ((a->valuestring == NULL) || (b->valuestring == NULL))
            {
                return false;
            }
            if (strcmp(a->valuestring, b->valuestring) == 0)
            {
                return true;
            }

            return false;

        case cJSON_Array:
        {
            cJSON *a_element = a->child;
            cJSON *b_element = b->child;

            for (; (a_element != NULL) && (b_element != NULL);)
            {
                if (!cJSON_Compare(a_element, b_element, case_sensitive))
                {
                    return false;
                }

                a_element = a_element->next;
                b_element = b_element->next;
            }

            /* one array is longer than the other */
            if (a_element != b_element) {
                return false;
            }

            return true;
        }

        case cJSON_Object:
        {
            cJSON *a_element = NULL;
            cJSON *b_element = NULL;
            cJSON_ArrayForEach(a_element, a)
            {
                /* TODO This has O(n^2) runtime, which is horrible! */
                b_element = get_object_item(b, a_element->string, case_sensitive);
                if (b_element == NULL)
                {
                    return false;
                }
                if (!cJSON_Compare(a_element, b_element, case_sensitive))
                {
                    return false;
                }
            }

            /* doing this twice, once on a and b to prevent one being a subset of the other */
            cJSON_ArrayForEach(b_element, b)
            {
                a_element = get_object_item(a, b_element->string, case_sensitive);
                if (a_element == NULL)
                {
                    return false;
                }
            }

            return true;
        }

        default:
            return false;
    }
}

cJSON_API void cJSON_Minify(char *json)
{
    char *into = json;
    while (*json)
    {
        if (*json == ' ')
        {
            json++;
        }
        else if (*json == '\t')
        {
            /* Whitespace characters. */
            json++;
        }
        else if (*json == '\r')
        {
            json++;
        }
        else if (*json=='\n')
        {
            json++;
        }
        else if ((*json == '/') && (json[1] == '/'))
        {
            /* double-slash comments, to end of line. */
            while (*json && (*json != '\n'))
            {
                json++;
            }
        }
        else if ((*json == '/') && (json[1] == '*'))
        {
            /* C-style comments. */
            while (*json && !((*json == '*') && (json[1] == '/')))
            {
                json++;
            }
            json += 2;
        }
        else if (*json == '\"')
        {
            /* string literals, which are \" sensitive. */
            *into++ = *json++;
            while (*json && (*json != '\"'))
            {
                if (*json == '\\')
                {
                    *into++ = *json++;
                }
                *into++ = *json++;
            }
            *into++ = *json++;
        }
        else
        {
            /* All other characters. */
            *into++ = *json++;
        }
    }
    *into = '\0'; /* and null-terminate. */
}

cJSON_API cJSON_bool cJSON_IsInvalid(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_Invalid;
}

cJSON_API cJSON_bool cJSON_IsFalse(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_False;
}

cJSON_API cJSON_bool cJSON_IsTrue(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_True;
}


cJSON_API cJSON_bool cJSON_IsBool(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & (cJSON_True | cJSON_False)) != 0;
}
cJSON_API cJSON_bool cJSON_IsNull(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_NULL;
}

cJSON_API cJSON_bool cJSON_IsNumber(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_Number;
}

cJSON_API cJSON_bool cJSON_IsString(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_String;
}

cJSON_API cJSON_bool cJSON_IsArray(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_Array;
}

cJSON_API cJSON_bool cJSON_IsObject(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_Object;
}

cJSON_API cJSON_bool cJSON_IsRaw(const cJSON * const item)
{
    if (item == NULL)
    {
        return false;
    }
    return (item->type & 0xFF) == cJSON_Raw;
}

cJSON_API void *cJSON_malloc(size_t size)
{
    return global_hooks.allocate(size);
}

cJSON_API void cJSON_free(void *object)
{
    global_hooks.deallocate(object);
}
