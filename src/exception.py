import sys
import logging

def error_msg_detail(error, error_detail:sys):
    _, _, tb = error_detail.exc_info()
    file_name = tb.tb_frame.f_code.co_filename
    error_messege = f"Error: {error} \nFile: {file_name} \nLine: {tb.tb_lineno}"
    
    return error_messege
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_msg_detail(error_message, error_detail=error_detail)
        
    def __str__(self):
        return self.error_message
        
        