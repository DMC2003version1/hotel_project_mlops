import traceback
import sys

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: Exception = None):
        super().__init__(error_message)
        self.error_message = CustomException.get_detailed_error_message(error_message=error_message, error_detail=error_detail)

    @staticmethod
    def get_detailed_error_message(error_message: str, error_detail: Exception = None) -> str:
        if error_detail is not None:
            try:
                _, _, exc_tb = sys.exc_info()
                if exc_tb:
                    line_number = exc_tb.tb_lineno
                    file_name = exc_tb.tb_frame.f_code.co_filename
                    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{error_message}]"
            except:
                error_message = f"Error message: [{error_message}]"
        else:
            error_message = f"Error message: [{error_message}]"
        return error_message
    
    def __str__(self):
        return self.error_message
    
    def __repr__(self) -> str:
        return f"CustomException({self.error_message})"
    
    
