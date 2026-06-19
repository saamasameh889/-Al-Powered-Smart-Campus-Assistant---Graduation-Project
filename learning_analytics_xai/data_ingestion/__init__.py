from .classroom_importer import (
    ClassroomImporter,
    CourseRecord,
    ComponentRecord,
    build_oauth_flow,
    exchange_code_for_token,
    CLASSROOM_SCOPES,
    _GOOGLE_AVAILABLE,
    get_coursework_status,
    get_pass_status,          # backward-compat alias for get_coursework_status
)
from .student_profile_builder import StudentProfileBuilder, StudentXAIProfile

__all__ = [
    "ClassroomImporter",
    "CourseRecord",
    "ComponentRecord",
    "StudentProfileBuilder",
    "StudentXAIProfile",
    "build_oauth_flow",
    "exchange_code_for_token",
    "CLASSROOM_SCOPES",
    "get_coursework_status",
    "get_pass_status",
]
