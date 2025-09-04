import sys
from src.loggers import logging

def error_message_details(error,error_detail :sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    error_msg = str(error)

    error_message = "The error file name is [{0}],the error line number is [{1}],the error message is [{2}]".format(file_name,line_num,error_msg)
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail=error_details)

    def __str__(self):
        return self.error_message