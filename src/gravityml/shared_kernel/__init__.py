"""Cross-cutting primitives shared by every bounded context."""

from gravityml.shared_kernel.error import ErrorInfo, fmap_error
from gravityml.shared_kernel.option import Nothing, Option, Some, from_optional
from gravityml.shared_kernel.result import Err, Ok, Result, UnwrapError
from gravityml.shared_kernel.safe import safe
from gravityml.shared_kernel.settings import Settings

__all__ = [
    "Err",
    "ErrorInfo",
    "Nothing",
    "Ok",
    "Option",
    "Result",
    "Settings",
    "Some",
    "UnwrapError",
    "from_optional",
    "fmap_error",
    "safe",
]
