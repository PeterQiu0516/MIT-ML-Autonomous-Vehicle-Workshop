class Tracker():
    def __init__(self, nhistory):
        self.history = nhistory
        self.buffer = [None]*nhistory
        #print self.buffer

    def shift_buffer(self):
        for i in range(self.history-1,0,-1):
            self.buffer[i] = self.buffer[i-1]
        self.buffer[0] = []

    def new_data(self, data):
        self.shift_buffer()
        self.buffer[0] = data

    def combined_results(self):
        # max occurence of item in buffer
        res = max(self.buffer,key=self.buffer.count)
        return res

