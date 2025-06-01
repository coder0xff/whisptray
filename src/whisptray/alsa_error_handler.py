"""Contains the ALSA error handler, which just logs the errors."""

import ctypes
import glob
import logging
import os
from sys import platform

# --- ALSA Error Handling Setup ---
# Define the Python callback function signature for ctypes
# Corresponds to:
# typedef void (*python_callback_func_t)(
#     const char *file,
#     int line,
#     const char *function,
#     int err,
#     const char *formatted_msg
# );
PYTHON_ALSA_ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(
    None,  # Return type: void
    ctypes.c_char_p,  # const char *file
    ctypes.c_int,  # int line
    ctypes.c_char_p,  # const char *function
    ctypes.c_int,  # int err
    ctypes.c_char_p,  # const char *formatted_msg
)

alsa_logger = logging.getLogger("ALSA")


def python_alsa_error_handler(file_ptr, line, func_ptr, err, formatted_msg_ptr):
    """
    Python callback to handle ALSA error messages passed from C.
    Decodes char* to Python strings.
    """
    try:
        file = (
            ctypes.string_at(file_ptr).decode("utf-8", "replace")
            if file_ptr
            else "UnknownFile"
        )
        function = (
            ctypes.string_at(func_ptr).decode("utf-8", "replace")
            if func_ptr
            else "UnknownFunction"
        )
        formatted_msg = (
            ctypes.string_at(formatted_msg_ptr).decode("utf-8", "replace")
            if formatted_msg_ptr
            else ""
        )

        # Using python logging to output ALSA messages
        alsa_logger.info(
            "%s:%d (%s) - err %d: %s", file, line, function, err, formatted_msg
        )
    except (UnicodeDecodeError, AttributeError, TypeError, ValueError) as e:
        # Fallback logging if there's an error within the error handler itself
        print(f"Error in python_alsa_error_handler: {e}")


# Keep a reference to the ctype function object to prevent garbage collection
py_error_handler_ctype = PYTHON_ALSA_ERROR_HANDLER_FUNC(python_alsa_error_handler)


def _load_alsa_redirect_lib():
    """
    Finds and loads the alsa_redirect.so C library.
    Returns the library object or None if not found/loaded.
    """
    c_redirect_lib = None
    # Try to load the C library for redirecting ALSA messages
    # Path when installed or running from source with Makefile-built .so
    c_redirect_lib_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "alsa_redirect.so"
    )

    if os.path.exists(c_redirect_lib_path):
        try:
            c_redirect_lib = ctypes.CDLL(c_redirect_lib_path)
            logging.info("Loaded alsa_redirect.so from: %s", c_redirect_lib_path)
            return c_redirect_lib
        except OSError as e:
            logging.error(
                "Error loading alsa_redirect.so from %s: %s", c_redirect_lib_path, e
            )
    else:
        # Fallback for when installed as a package, where setuptools renames the .so
        package_dir = os.path.dirname(os.path.abspath(__file__))
        found_libs = list(glob.glob(os.path.join(package_dir, "alsa_redirect*.so")))
        if found_libs:
            # Take the first one found (should ideally be only one)
            # Sort to get a deterministic choice if multiple somehow exist
            found_libs.sort()
            c_redirect_lib_path_found = found_libs[0]
            try:
                c_redirect_lib = ctypes.CDLL(c_redirect_lib_path_found)
                logging.info(
                    "Loaded compiled C extension: %s", c_redirect_lib_path_found
                )
                return c_redirect_lib
            except OSError as e:
                logging.error(
                    "Error loading compiled C extension %s: %s",
                    c_redirect_lib_path_found,
                    e,
                )
        else:
            # Last resort: try loading from system path (less reliable)
            try:
                c_redirect_lib = ctypes.CDLL("alsa_redirect.so")
                logging.info("Loaded alsa_redirect.so from system path.")
                return c_redirect_lib
            except OSError:
                logging.error(
                    "alsa_redirect.so not found at %s, nor as alsa_redirect*.so in"
                    " package dir, nor in system paths.",
                    c_redirect_lib_path,
                )
    return None


def _define_c_lib_interfaces(c_lib):
    """Defines the C function interfaces for the ALSA redirect library."""
    if not c_lib:
        logging.error("C library object is None, cannot define interfaces.")
        return

    try:
        # void register_python_alsa_callback(python_callback_func_t callback);
        c_lib.register_python_alsa_callback.argtypes = [PYTHON_ALSA_ERROR_HANDLER_FUNC]
        c_lib.register_python_alsa_callback.restype = None

        # int initialize_alsa_error_handling();
        c_lib.initialize_alsa_error_handling.argtypes = []
        c_lib.initialize_alsa_error_handling.restype = ctypes.c_int

        # int clear_alsa_error_handling();
        c_lib.clear_alsa_error_handling.argtypes = []
        c_lib.clear_alsa_error_handling.restype = ctypes.c_int
        logging.info("Successfully defined C library interfaces.")
    except AttributeError as e:
        logging.error(
            "Error defining C library interfaces: %s. Library might not have expected"
            " functions.",
            e,
        )


def _get_and_prepare_alsa_lib():
    """Loads the ALSA C library and defines its interfaces."""
    c_lib = _load_alsa_redirect_lib()
    if c_lib:
        _define_c_lib_interfaces(c_lib)
        # Check if critical functions are actually defined after attempting to define
        # interfaces. This is a basic check. More robust checks could verify specific
        # function pointers.
        if (
            not hasattr(c_lib, "initialize_alsa_error_handling")
            or not hasattr(c_lib, "clear_alsa_error_handling")
            or not hasattr(c_lib, "register_python_alsa_callback")
        ):
            logging.error(
                "Critical C library functions not found after defining interfaces."
            )
            return None
    return c_lib


def setup_alsa_error_handler():
    """
    Sets up a custom ALSA error handler using the C helper library.
    """
    if "linux" not in platform:
        logging.info("Skipping ALSA error handler setup on non-Linux platform.")
        return

    try:
        c_redirect_lib = _get_and_prepare_alsa_lib()
        if c_redirect_lib is None:
            logging.error(
                "Failed to load/prepare c_redirect_lib. Cannot set ALSA error handler."
            )
            return

        c_redirect_lib.register_python_alsa_callback(py_error_handler_ctype)
        logging.info("Registered Python ALSA error handler with C helper.")

        ret = c_redirect_lib.initialize_alsa_error_handling()
        if ret < 0:
            logging.error(
                "C library failed to set ALSA error handler. Error code: %d", ret
            )

    except (OSError, AttributeError, TypeError, ValueError, ctypes.ArgumentError) as e:
        logging.error("Error setting up ALSA error handler: %s", e, exc_info=True)


def teardown_alsa_error_handler():
    """
    Clears the custom ALSA error handler using the C helper library.
    """
    if "linux" not in platform:
        logging.info("Skipping ALSA error handler teardown on non-Linux platform.")
        return

    try:
        c_redirect_lib = _get_and_prepare_alsa_lib()
        if c_redirect_lib is None:
            logging.error(
                "Failed to load/prepare c_redirect_lib for teardown. Cannot clear ALSA"
                " error handler."
            )
            return

        ret = c_redirect_lib.clear_alsa_error_handling()
        if ret < 0:
            logging.error(
                "C library failed to clear ALSA error handler. Error code: %d", ret
            )
        else:
            logging.info("Successfully cleared ALSA error handler via C helper.")

    except (OSError, AttributeError, TypeError, ValueError, ctypes.ArgumentError) as e:
        logging.error("Error during ALSA error handler teardown: %s", e, exc_info=True)
