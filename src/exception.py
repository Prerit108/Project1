import sys  #commonly used to access runtime details like the current exception traceback.
from src.logger import  logging



def error_message_detail(error,error_detail:sys):      ## error_detail should be of type/module sys
    _,_,exc_tb = error_detail.exc_info()  ## skipping first 2 info (not useful)
    file_name = exc_tb.tb_frame.f_code.co_filename  ## given in custom exception handling documentation
    error_message = "Error occured in python script name[{0}] line number [{1}] error message[{2}]".format(
    file_name,exc_tb.tb_lineno,str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    

    ## Checking if it is working
"""if __name__ == "__main__":
    try:
        a =1/0
    except Exception as ex:
        err = CustomException(ex,sys)
        logging.info(err)
        raise err from None""" ## from None will only show your custom exception, not the original ZeroDivisionError in the terminal


