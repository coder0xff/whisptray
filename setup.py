import subprocess
from setuptools import Extension, setup


def get_pkg_config_flags(package_name, option):
    """Helper to get flags from pkg-config."""
    try:
        process = subprocess.run(
            ['pkg-config', option, package_name],
            capture_output=True,
            text=True,
            check=True
        )
        output_list = process.stdout.strip().split()
        if option == '--libs-only-l':
            # e.g., turn ['-lasound'] into ['asound']
            return [lib[2:] for lib in output_list if lib.startswith('-l')]
        if option == '--cflags-only-I':
            # e.g., turn ['-I/usr/include/alsa'] into ['/usr/include/alsa']
            return [p[2:] for p in output_list if p.startswith('-I')]
        if option == '--libs-only-L':
            # e.g., turn ['-L/usr/lib'] into ['/usr/lib']
            return [p[2:] for p in output_list if p.startswith('-L')]
        return output_list # For other generic flags if any
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback if pkg-config or the package is not found
        if package_name == "alsa":
            if option == "--libs-only-l":
                return ["asound"]
            if option == "--libs-only-L":
                return []
            if option == "--cflags-only-I":
                return []
        print(f"Warning: pkg-config could not find {package_name} with option {option}. Using fallback: {package_name=}, {option=}")
        return []

alsa_redirect_extension = Extension(
    'whisptray.alsa_redirect',
    sources=['src/alsa_redirect.c'],
    include_dirs=get_pkg_config_flags('alsa', '--cflags-only-I'),
    library_dirs=get_pkg_config_flags('alsa', '--libs-only-L'),
    libraries=get_pkg_config_flags('alsa', '--libs-only-l'),
    extra_compile_args=['-O2', '-g']
)


setup(
    ext_modules=[alsa_redirect_extension]
    # packages, package_dir, and zip_safe are handled by pyproject.toml
) 