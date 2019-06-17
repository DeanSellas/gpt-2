import os

class pyLogger():
    def __init__(self, path=None, fileName="log.txt"):
        if path is None:
            self.path = os.getcwd()
            try:
                self.path = os.path.join(self.path, "logs")
            except:
                os.mkdir("logs")
                self.path = os.path.join(self.path, "logs")
        else:
            self.path = path

        self.fileName = fileName
        filePath = os.path.join(self.path, self.fileName)

        # if file size is larger than 20mb rename it and start new log file
        if os.path.getsize(filePath) > 20000000:
            os.rename(filePath, os.path.join(self.path, "Old_Log.txt"))

        
        self.outputArr = list()


    def _print(self, m, t='INFO'):
        """
        Alternative for printing to the console. Allows for the output to be displayed to the log.

        m variable is the message you want to display

        t is the prefix to the message. Can be used to organize the output into different types 
        """
        if t != None:
            print("{}: {}".format(t, m))
            self.outputArr.append("{}: {}\n".format(t, m))
        else:
            print("{}".format(m))
            self.outputArr.append("{}\n".format(m))

    def _save(self, fileName = None):
        self.outputArr.append("\n\n")
        
        # did this so custom outputs can be saved separate from the log file
        if fileName == None:
            fileName = self.fileName

        self.logFile = open("{}\\{}".format(self.path, fileName), 'a')
        self.logFile.writelines(self.outputArr)
        
        self.outputArr.clear()


    def close(self):
        self._save()
        self.logFile.close()
        