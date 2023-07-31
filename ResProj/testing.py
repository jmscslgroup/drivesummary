from PyQt5 import QtWidgets
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QVBoxLayout, QSpacerItem, QPushButton
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as Navi
from matplotlib.figure import Figure
import pandas as pd
from PyQt5 import sip as sip
import numpy as np
from numpy import gradient
from visionFinalPROJECT import Ui_MainWindow
from strym import strymread
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from scipy import integrate
import strym
import cantools
import traceback, sys
import traces
from dateutil.parser import parse as date_parse

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        
        #self.kwargs['progress_callback'] = self.signals.progress
    
    @pyqtSlot()
    def run(self):
        '''
        Executes function with given ags, kwags
        '''
        try:
            print("Worker", int(QThread.currentThread().currentThreadId()))
            print("Running worker function pre:")
            result = self.fn(*self.args, **self.kwargs)
            print("Running worker function post:")
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done

class MatplotlibCanvas(FigureCanvasQTAgg):      
    def __init__(self,parent=None, dpi = 120):
        fig = Figure(dpi = dpi)
        fig.set_facecolor('#09111c')
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas,self).__init__(fig)
        #fig.tight_layout() #  uncomment for tight layout

class Main_Window(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        #self.setStyleSheet("background-color: grey;")

        self.setupUi(self)

        self.filename = ''
        self.canv = MatplotlibCanvas(self)
        self.df = []
        self.speedDF = []
        self.steeringAngleDF = []
        self.steeringRateDF = []
        self.cruiseDF = []
        self.dataName = ''
        self.db=''
        self.verifyDfs = False
        self.characteristics = {}
        self.currentLayout = ''
        self.dbcfile = 'dbc\\nissan_rogue_2021.dbc'

        self.toolbar = Navi(self.canv,self.centralwidget)
        #self.horizontalLayout.addWidget(self.toolbar)

        self.themes = ['bmh', 'classic', 'dark_background', 'fast', 
        'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-bright',
         'seaborn-colorblind', 'seaborn-dark-palette', 'seaborn-dark', 
         'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook',
         'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk',
         'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'seaborn',
         'Solarize_Light2', 'tableau-colorblind10']
        
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
        print("Main app:", int(QThread.currentThread().currentThreadId()))

        self.actionOpen.triggered.connect(self.getFile)
    
    def Update(self):
        # self.currentLayout = self.stackedWidget.currentWidget().findChild(QVBoxLayout)
        # Legacy function above. To update graph widget is as follows


        
        #  Below functions are for themes
        #print("Value from Combo Box:",value)
        #plt.clf()
        #plt.style.use(value)
       
       
        #try:
        self.compileCharacteristics()

        self.ridedistancelabel.setText("Total Distance (Meters): " + str(np.round(self.characteristics['tripDistance'], 2)))
        self.ridetimelabel.setText("Total Time (Minutes): " + str(np.round(self.characteristics['tripTime'], 2)))
        self.cctimelabel.setText("Cruise Time(Minutes): " + str(np.round(self.characteristics['cruiseTime'], 2)))
        self.ccpercentlabel.setText("Percent in CC: " + str(np.round(self.characteristics['cruisePercentage'], 2)) + "%")
        self.stopslabel.setText("Num Stops: " + str(self.characteristics['tripStops']))
        '''
            #self.cruiseDF = self.df.acc_state(plot = True)
        except Exception as error:
            print("An exception occurred with the characteristics: ", error)

        t0 = self.cruiseDF.iloc[0].Time
        plt.plot(self.cruiseDF.Time-t0,self.cruiseDF.Message)
        '''

        
        try:
            self.canv = MatplotlibCanvas(self)
            self.toolbar = Navi(self.canv,self.centralwidget)
            layout = QtWidgets.QVBoxLayout()
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canv)

            self.graphwidget.setLayout(layout)

            self.canv.axes.cla()
            ax = self.canv.axes

            
            self.speedDF = self.dateIndexConversion(self.speedDF)
            self.speedDF.plot(x = 'Time', y = 'Message', ax = ax)
            #ax.plot(self.speedDF['Clock'],self.speedDF.Message)
            ax.tick_params(labelsize=8, colors='white')
            ax.set_ylabel('Speed (Km/H)', fontweight ='bold', color='white')
            ax.set_title('Time (H:M:S) vs Speed', fontweight ='bold', color='white')
            legend = ax.legend()
            legend.set_draggable(True)

            self.canv.draw()

        except Exception as error:

            print("An exception occurred with the graph: ", error)




        #graph = strymread.plt_ts(self.speedDF, "Speed", ax = self.canv.axes)

        #self.visualizeData(self.df.speed())

        #self.annot_max(self.df.index, self.df[self.dataName],ax)

    def compileCharacteristics(self):
        #accelComfort
        #self.accelComfort()
        
        
        #tripDistance
        tripDistance = self.tripDistance()
        
        
        #tripStops
        percentageHisto = self.cruiseTime()
        cruisePercentage = (percentageHisto[1] * 100)
        
        tripStops = self.tripStops()
        #tripTIme
        tripTimeSeconds = self.speedDF["Time"].iloc[-1] - self.speedDF["Time"].iloc[0]
        tripTime = tripTimeSeconds/60

        cruiseTime = tripTime * percentageHisto[1]

        self.characteristics = {'tripDistance': tripDistance, 'tripTime': tripTime, 'cruisePercentage': cruisePercentage,
                                'cruiseTime': cruiseTime, 'tripStops': tripStops}

        #self.characteristics = {'tripDistance': tripDistance, 'numberStops': numStops}
        #make a dictionary here. Tripdistance is one thing

    def accelComfort(self):
        self.speedDF['Accel'] = (self.speedDF['Message'] - self.speedDF['Message'].shift(1)) / (self.speedDF['Time'] - self.speedDF['Time'].shift(1)).fillna(0)

        print(self.speedDF['Accel'])

        
        
        
        #sum((accelDF.Time - accelDF.Time.shift(1)) if accelDF.Message > 30)

    def cruiseTime(self):
        #convert to datetime
        self.cruiseDF = self.dateIndexConversion(self.cruiseDF)
        idx = self.cruiseDF.Time
        status = self.cruiseDF.Message


        plt.plot(self.cruiseDF.Time,self.cruiseDF.Message)
        plt.show()

        ts = traces.TimeSeries(default=0)
        idx = self.cruiseDF.Time.dt.strftime('%Y-%m-%d %H:%M:%S')
        for date_string, status_value in zip(idx, status):
            ts[date_parse(date_string)] = status_value

        print(ts.distribution(
            start=date_parse(str(self.cruiseDF["Time"].iloc[0])),
            end=date_parse(str(self.cruiseDF["Time"].iloc[-1]))))

        # compute distribution  
        return ts.distribution(
            start=date_parse(str(self.cruiseDF["Time"].iloc[0])),
            end=date_parse(str(self.cruiseDF["Time"].iloc[-1])))



        '''
        temp = self.cruiseDF.asfreq(freq='1S')
        print(temp.to_string())
        temp2 = temp['Message'].value_counts()[1]
        print(temp2)
        '''

        '''
        #turn to pivot table
        df1 = self.cruiseDF.pivot_table(index=[self.cruiseDF.Time.dt.date],
                            columns='Message', values='Time', aggfunc='first')
        
        df1[1] = df1[1].fillna(pd.to_datetime(df1.index.to_series()))
        df1[0] = df1[0].fillna(pd.to_datetime(
        df1.index.to_series())+ pd.DateOffset(+1))

        print(df1[0].asfreq('1D').fillna(pd.Timedelta(seconds=0)))
        
        result = (df1[1] - df1[0]).asfreq('1D').fillna(pd.Timedelta(seconds=0))
        result0 = (df1[0] - df1[1]).asfreq('1D').fillna(pd.Timedelta(seconds=0))

        print(abs(result0[0]))
        print(result[0])
        '''
        #df1[True] = df1[True].fillna(pd.to_datetime(df1.index.to_series()))
        #df1[False] = df1[False].fillna(pd.to_datetime(
            #df1.index.to_series()) + pd.DateOffset(+1))
        #result = (df1[False] - df1[True]).asfreq('1D').fillna(pd.Timedelta(seconds=0))
    def tripStops(self):
        g = self.speedDF['Message'].ne(self.speedDF['Message'].shift()).cumsum()
        g = g[self.speedDF['Message'].eq(0)]

        return g.groupby(g).count().ge(3).sum()

    def tripDistance(self, time = -1):
        speed  = self.speedDF
        speed_ms_conv =pd.DataFrame()
        speed_ms_conv['Time'] = speed['Time']
        speed_ms_conv['Message'] = speed['Message']*1000.0/3600.0

        dist = integrate.cumtrapz(speed_ms_conv['Message'], speed_ms_conv['Time'].values, initial = 0.0)

        distance = pd.DataFrame()
        distance['Time'] = speed_ms_conv['Time']
        distance['Message'] = dist

        required_distance = 0.0
        if time == -1:
            required_distance =  distance['Message'].iloc[-1]
        else:
            if time <= self.triptime():
                desired_time = dist['Time'].iloc[0] + time
                data = dist[dist['Time'] > desired_time]
                required_distance =  data['Message'].iloc[0]
        return required_distance


    def getFile(self):
        """ This function will get the address of the csv file location
            also calls a readData function 
        """
        # for multiple
        # self.filename = QFileDialog.getOpenFileName(filter = "csv (*.csv)")[0]  - legacy function
        # print("File :", self.filename)
        print("Get function executed, initial.")
        self.file = QFileDialog.getOpenFileName(filter = "csv (*.csv)")[0]

        print("Queing workers, pre")
        worker = Worker(self.readData, self.file, self.dbcfile)
        worker.signals.finished.connect(self.Update)
        worker2 = Worker(self.strymRead, self.file, self.dbcfile)
        worker2.signals.finished.connect(self.Update)
        self.threadpool.start(worker)
        #self.threadpool.start(worker2)
        print("Workers queued, post")
    
    def strymRead(self, caninput, dbcinput):
        self.df = strymread(csvfile=caninput, dbcfile=dbcinput)

    def dateIndexConversion(self, messageDF):
        
        #removing bus column, unneccesary for plotting

        if 'Bus' in messageDF.columns:
            newDF = messageDF.drop(columns=['Bus'])
        
        #converting Time column to datetime and assigning as variable
        
        Time = pd.to_datetime(newDF['Time'], unit='s')
        
        #resetting the index and changing it to date/time
        
        #newDF.reset_index(drop=True, inplace=True)
        newDF['Time'] = pd.DatetimeIndex(Time).tolist()


        #Now removing duplicates
        newDF = newDF[~messageDF.Time.duplicated(keep='first')]

        return newDF

    def readData(self, caninput, dbcinput):

        self.db = cantools.database.Database()
        with open(dbcinput,'r') as fin:    
            self.db = cantools.database.load(fin)
        self.df = pd.read_csv(caninput)


        #speed and speed date dfs
        self.speedDF = strym.convertData('SPEED','SPEED',self.df,self.db)#kph
        self.speedDF.to_csv("Example.csv")
        self.cruiseDF = strym.convertData('CRUISE','CRUISE_ENGAGED',self.df,self.db)

       
def main():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    w = Main_Window()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__': main()


"""
dataframe = strymread(csvfile='C:\\Users\\flame\\Downloads\\CSV Dfs\\TotalSmaller.csv', dbcfile="nissan_rogue_2021.dbc")
print('done')



speedDF = dataframe.speed()

speedDF.plot(legend=True)



strymread.plt_ts(speedDF, "Wut is this")


print('done2')


cruiseDF = dataframe.acc_state()
steeringAngleDF = dataframe.msg_subset(conditions = steering_angle)


"""