"""
Enhanced exception classes for VNE CLI.
"""

class VNECLIError(Exception):
    """Base exception for CLI-related errors."""
    
    def __init__(self, message: str, details: str = None, exit_code: int = 1):
        super().__init__(message)
        self.message = message
        self.details = details
        self.exit_code = exit_code

class CommandError(VNECLIError):
    """Exception raised when command execution fails."""
    pass

class ValidationError(VNECLIError):
    """Exception raised for validation errors."""
    pass

class FileError(VNECLIError):
    """Exception raised for file operation errors."""
    pass

class AlgorithmError(VNECLIError):
    """Exception raised for algorithm-related errors."""
    pass
