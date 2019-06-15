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
        self.outputArr = list()


    def _print(self, m, t='INFO'):
        if t != None:
            print("{}: {}".format(t, m))
            self.outputArr.append("{}: {}\n".format(t, m))
        else:
            print("{}".format(m))
            self.outputArr.append("{}\n".format(m))

    def _save(self):
        self.outputArr.append("\n\n")
        self.logFile = open("{}\\{}".format(self.path, self.fileName), 'a')
        self.logFile.writelines(self.outputArr)
        
        self.outputArr.clear()


    def close(self):
        self._save()
        self.logFile.close()
        