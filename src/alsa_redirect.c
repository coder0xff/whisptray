// alsa_redirect.c
#include <stdio.h>
#include <stdarg.h>
#include <string.h> // For strerror
#include <alsa/asoundlib.h>

// Define the type for the Python callback
// It will receive the already formatted message, plus original context
typedef void (*python_callback_func_t)(const char *file, int line, const char *function, int err, const char *formatted_msg);

static python_callback_func_t user_python_callback = NULL;

// C function that Python will call to register its callback
// This function needs to be exported from the shared library.
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT void register_python_alsa_callback(python_callback_func_t callback) {
    user_python_callback = callback;
}

// This is our custom ALSA error handler
// It's not directly exported for Python to call, but used by snd_lib_error_set_handler
void alsa_c_error_handler(const char *file, int line, const char *function, int err, const char *fmt, ...) {
    char buffer[2048];
    va_list args;

    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer) - 1, fmt, args);
    buffer[sizeof(buffer) - 1] = '\0';
    va_end(args);

    if (user_python_callback) {
        user_python_callback(file, line, function, err, buffer);
    } else {
        fprintf(stderr, "ALSA lib (C fallback) %s:%d:(%s) ", file ? file : "?", line, function ? function : "?");
        fprintf(stderr, "%s", buffer);
        if (err != 0) {
            fprintf(stderr, ": %s", snd_strerror(err));
        }
        putc('\n', stderr);
    }
}

// Python calls this function to ask our C code to set up the ALSA error handler.
EXPORT int initialize_alsa_error_handling() {
    if (snd_lib_error_set_handler(&alsa_c_error_handler) < 0) {
        // Optionally, print an error here if Python side isn't robust enough yet
        // fprintf(stderr, "C: Failed to set ALSA error handler.\n");
        return -1; // Indicate failure
    }
    return 0; // Indicate success
}

// Optional: A function to revert to the default ALSA handler from C
EXPORT int clear_alsa_error_handling() {
    if (snd_lib_error_set_handler(NULL) < 0) {
        return -1;
    }
    return 0;
}
