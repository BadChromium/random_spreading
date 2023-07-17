import logging
import logging.handlers
import os.path
import sys
import time

LOG_FORMAT = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %p"

def empty_function(*args, **kargs):
    pass

logging.basicConfig = empty_function
logging.raiseExceptions = True

class Logger:
    def __init__(self, name='Logger', loglevel='WARNING', save=True, save_level=logging.WARNING, log_path=None, update_frequency='D', save_interval=1):
        '''
        Initialize the self-defined class Logger
        name: The logger's name
        loglevel: The log level which will be output through terminal
        save: Save to files or not
        save_level: The log level of saved logs
        update_frequency: The frequency of log update (For log type 3, see below)
        save_interval: The interval of savings
        '''
        self.log_path = log_path
        self.name = name
        self.loglevel = loglevel # Strings for input
        self.log_level = None # Choose log level based on the string above
        self.save = save
        self.save_level = save_level
        self.logging = logging
        self.log_format = LOG_FORMAT
        self.date_format = DATE_FORMAT
        self.update_frequency = update_frequency
        self.save_interval = save_interval
    
    def init_streamHdl(self, formatter):
        streamHandler = self.logging.StreamHandler(sys.stdout)
        streamHandler.setLevel(self.log_level)
        streamHandler.setFormatter(formatter)
        return streamHandler
    
    def init_fileHdl(self, formatter, log_file_name='log.log'):
        fileHandler = self.logging.FileHandler(log_file_name)
        fileHandler.setLevel(self.save_level)
        fileHandler.setFormatter(formatter)
        return fileHandler
    
    def init_rotatingfileHdl(self, formatter, log_file_name='log.log'):
        '''
        formatter: For formatting the logs
        log_file_name: The name of saved log files
        '''
        fh = self.logging.handlers.RotatingFileHandler(log_file_name, maxBytes=1024*1024*10, backupCount=5, mode='a')
        fh.setLevel(self.save_level)
        fh.setFormatter(formatter)
        return fh
    
    def init_timedRotatingfileHdl(self, formatter, log_file_name='log.log'):
        fh = self.logging.handlers.TimedRotatingFileHandler(log_file_name, when=self.update_frequency, interval=self.save_interval, backupCount=5)
        fh.setLevel(self.save_level)
        fh.setFormatter(formatter)
        return fh
    
    def init_logger(self, fh_type='1'):
        '''
        :param: fh_type: The file handler's type. 1 for common, 2 for rotating file handler, 3 for time rotating
        :return: The self-defined logger
        '''
        if self.loglevel == "DEBUG":
            self.log_level = self.logging.DEBUG
        elif self.loglevel == "INFO":
            self.log_level = self.logging.INFO
        elif self.loglevel == "WARNING":
            self.log_level = self.logging.WARNING
        elif self.loglevel == "ERROR":
            self.log_level = self.logging.ERROR
        else:
            self.log_level = self.logging.CRITICAL

        # Create a logger
        logger = self.logging.getLogger(self.name)
        logger.setLevel(self.log_level)
        formatter = self.logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        logger.addHandler(self.init_streamHdl(formatter))

        if self.save:
            log_file = os.path.join(self.log_path, self.name + '.log')
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            if not os.path.exists(log_file):
                os.system(r"touch {}".format(log_file))
            if fh_type == '1':
                fh = self.init_fileHdl(formatter, log_file)
                logger.addHandler(fh)
            elif fh_type == '2':
                fh = self.init_rotatingfileHdl(formatter, log_file)
                logger.addHandler(fh)
            elif fh_type == '3':
                fh = self.init_timedRotatingfileHdl(formatter, log_file)
                logger.addHandler(fh)
        
        return logger
        
if __name__ == '__main__':
    try:
        print(f"Number of params: {len(sys.argv)}")
        print(f"Params: {str(sys.argv)}")
        print(f"Script name: {sys.argv[0]}")
        for item in range(1, len(sys.argv)):
            print(f"Param {item}: {sys.argv[item]}")
    except Exception as e:
        print(e)
    mylogger = Logger(name='test', loglevel='WARNING', save=True, save_level=logging.DEBUG, log_path='./logs').init_logger(fh_type='3')
    mylogger.debug('debug message')
    mylogger.info('info message')
    mylogger.warning('warning message')