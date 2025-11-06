# src/dmp/config.py

# Global configuration variables for the DMP package

VERBOSE = False  # default value

def set_verbose(value: bool):
    """Set the global verbosity flag for the entire package."""
    global VERBOSE
    VERBOSE = value
