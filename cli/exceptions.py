"""
Exception classes for VNE CLI with error context.
"""


class VNECLIError(Exception):
    """Base exception for CLI-related errors."""
    
    def __init__(self, message: str, details: str = None, exit_code: int = 1, context: str = None):
        super().__init__(message)
        self.message = message
        self.details = details
        self.exit_code = exit_code
        self.context = context


class CommandError(VNECLIError):
    """Exception raised when command execution fails."""
    
    def __init__(self, message: str, command: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.command = command


class ValidationError(VNECLIError):
    """Exception raised for validation errors."""
    
    def __init__(self, message: str, field: str = None, value=None, **kwargs):
        super().__init__(message, **kwargs)
        self.field = field
        self.value = value


class FileError(VNECLIError):
    """Exception raised for file operation errors."""
    
    def __init__(self, message: str, filepath: str = None, operation: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.filepath = filepath
        self.operation = operation


class AlgorithmError(VNECLIError):
    """Exception raised for algorithm-related errors."""
    
    def __init__(self, message: str, algorithm: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.algorithm = algorithm
