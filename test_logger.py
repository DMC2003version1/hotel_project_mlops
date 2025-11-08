from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger()

def division(a, b):
    try:
        logger.info("Starting the program")
        c = a/b
    except Exception as e:
        raise CustomException(e, sys)
    
    
if __name__ == "__main__":
    try:
        division(1, 0)
    except CustomException as e:
        logger.error(str(e))
    
