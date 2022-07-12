import os
from os import listdir
from os.path import isfile, join
import sys
import numpy as np
from pandas import DataFrame as df
import re
import yaml
import struct
import ipywidgets as widgets
from ipywidgets import Button, Layout
from IPython.display import clear_output
from importlib import reload
import matplotlib.pyplot as plt
import cmath
import re
import yaml
from lmfit import model, Model
from lmfit.models import GaussianModel, SkewedGaussianModel, VoigtModel, ConstantModel, LinearModel, QuadraticModel, PolynomialModel

# Plotly settings
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
pio.renderers.default = 'notebook+plotly_mimetype'
pio.templates.default = 'simple_white'
pio.templates[pio.templates.default].layout.update(dict(
    title_y = 0.9,
    title_x = 0.5,
    title_xanchor = 'center',
    title_yanchor = 'top',
    legend_x = 0,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)'
))

##### Folders #####

Folders = {}
Folders['Parameters'] = os.getcwd()+'/Parameters'
Folders['Fits'] = os.getcwd()+'/Fits'
Folders['Figures'] = os.getcwd()+'/Figures'

##### Data Tools #####

class DataTools :
    
    def __init__(self) :
        
        pass
    
    def FileList(self,FolderPath,Filter) :
        
        os.makedirs(Folders['Parameters'], exist_ok=True)
        FileList = [f for f in listdir(FolderPath) if isfile(join(FolderPath, f))]
        for i in range(len(Filter)):
            FileList = [k for k in FileList if Filter[i] in k]
        for i in range(len(FileList)):
            FileList[i] = FileList[i].replace('.yaml','')
        
        return FileList
        
    def Load_TDS(self,Parameters) :
        
        FolderPath = Parameters['FolderPath']
        FileName = Parameters['FileName']
        Masses = Parameters['Masses']

        with open(FolderPath + '/' + FileName, mode='rb') as file:
            fileContent = file.read()

        NumChan = len(Masses) + 1
        DataLength = int((len(fileContent)-5)/(46*NumChan))
        Data = np.zeros((int(1+NumChan),DataLength))

        for i in range(len(Data)) :
            for j in range(len(Data[0])) :
                if i == 0 :
                    index = int(31+j*46*NumChan)
                    Data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]/1000
                else :
                    index = int(43+j*46*NumChan + (i-1)*46)
                    Data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]
        
        Header = list()
        Header.append('Time (s)')
        if 'TChan' in Parameters :
            TChan = Parameters['TChan']
        else :
            TChan = 0
        for idx in range(NumChan) :
            if idx == TChan :
                Header.append('Temperature (K)')
            else :
                Header.append('Mass '+str(Masses[idx-1]))
        
        Data = df(np.transpose(Data),columns=Header)
        Data = Data.set_index('Temperature (K)')
        if 'TScale' in Parameters :
            Data.index = Data.index * Parameters['TScale']
        
        Parameters['HeatingRate'] = np.mean(np.diff(Data.index)/np.diff(Data['Time (s)']))
            
        return Data, Parameters
    
    def RemoveEmptyDataSets(self,Data,Threshold) :
        
        Index = list()
        for i in Data.columns :
            if np.mean(Data[i]) < Threshold :
                Index.append(i)
        for i in Index :
            del Data[i]
        
        return Data
    
    def TrimData(self,Data,Min,Max) :
        
        Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
        Data = Data[Mask]
        
        return Data
    
    def ReduceResolution(self,Data,Resolution=1) :
        
        Counter = 0
        ReducedData = df()
        for i in range(int(len(Data.columns.values)/Resolution)) :
            Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
            ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
            Counter = Counter + Resolution
        
        return ReducedData

##### Fit Tools #####

class FitTools :
    
    def __init__(self,Data,FitInfo,Name='') :
        
        self.Data = Data
        self.FitInfo = FitInfo
        self.Name = Name
        
        try :
            FitInfo['ModelType']
            FitInfo['Models']
        except:
            ModelType = 'None'
            ModelString = ''
        else :
            if FitInfo['ModelType'] == 'BuiltIn' :
                self.BuiltInModels()
            if FitInfo['ModelType'] == 'SFG' :
                self.SFGModel()
    
    def BuiltInModels(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append((key,FitInfo['Models'][key]['model']))
        
        for Model in ModelString :
            try :
                FitModel
            except :
                if Model[1] == 'Constant' :
                    FitModel = ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = VoigtModel(prefix=Model[0]+'_')
            else :
                if Model[1] == 'Constant' :
                    FitModel = FitModel + ConstantModel(prefix=Model[0]+'_')
                if Model[1] == 'Linear' :
                    FitModel = FitModel + LinearModel(prefix=Model[0]+'_')
                if Model[1] == 'Gaussian' :
                    FitModel = FitModel + GaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'SkewedGaussian' :
                    FitModel = FitModel + SkewedGaussianModel(prefix=Model[0]+'_')
                if Model[1] == 'Voigt' :
                    FitModel = FitModel + VoigtModel(prefix=Model[0]+'_')
        
        self.FitModel = FitModel
        self.ModelParameters = FitModel.make_params()
        
    def SFGModel(self) :
        
        FitInfo = self.FitInfo
        
        ModelString = list()
        for key in FitInfo['Models'] :
            ModelString.append([key])
        
        if len(ModelString) == 2 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 3 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 4 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 5 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 6 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 7 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 8 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak7_amp*(cmath.exp(Peak7_phi*1j)/(x-Peak7_omega+Peak7_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        elif len(ModelString) == 9 :
            def SFGFunction(x,NonRes_amp,
                            Peak1_amp,Peak1_phi,Peak1_omega,Peak1_gamma,
                            Peak2_amp,Peak2_phi,Peak2_omega,Peak2_gamma,
                            Peak3_amp,Peak3_phi,Peak3_omega,Peak3_gamma,
                            Peak4_amp,Peak4_phi,Peak4_omega,Peak4_gamma,
                            Peak5_amp,Peak5_phi,Peak5_omega,Peak5_gamma,
                            Peak6_amp,Peak6_phi,Peak6_omega,Peak6_gamma,
                            Peak7_amp,Peak7_phi,Peak7_omega,Peak7_gamma,
                            Peak8_amp,Peak8_phi,Peak8_omega,Peak8_gamma) :
                Peaks = NonRes_amp
                Peaks+= Peak1_amp*(cmath.exp(Peak1_phi*1j)/(x-Peak1_omega+Peak1_gamma*1j))
                Peaks+= Peak2_amp*(cmath.exp(Peak2_phi*1j)/(x-Peak2_omega+Peak2_gamma*1j))
                Peaks+= Peak3_amp*(cmath.exp(Peak3_phi*1j)/(x-Peak3_omega+Peak3_gamma*1j))
                Peaks+= Peak4_amp*(cmath.exp(Peak4_phi*1j)/(x-Peak4_omega+Peak4_gamma*1j))
                Peaks+= Peak5_amp*(cmath.exp(Peak5_phi*1j)/(x-Peak5_omega+Peak5_gamma*1j))
                Peaks+= Peak6_amp*(cmath.exp(Peak6_phi*1j)/(x-Peak6_omega+Peak6_gamma*1j))
                Peaks+= Peak8_amp*(cmath.exp(Peak8_phi*1j)/(x-Peak8_omega+Peak8_gamma*1j))
                return np.real(Peaks*np.conjugate(Peaks))
        
        FitModel = Model(SFGFunction)
        ModelParameters = FitModel.make_params()
        
        self.FitModel = FitModel
        self.ModelParameters = ModelParameters
    
    def SetParameters(self, Value = None) :
        
        FitInfo = self.FitInfo
        ModelParameters = self.ModelParameters
        
        ParameterList = ['amp','phi','omega','gamma','center','sigma','c']
        Parameters = {'Standard': FitInfo['Models']}

        if 'Cases' in FitInfo and Value != None:
            for Case in FitInfo['Cases'] :
                if Value >= min(FitInfo['Cases'][Case]['zRange']) and Value <= max(FitInfo['Cases'][Case]['zRange']) :
                    Parameters[Case] = FitInfo['Cases'][Case]
        
        for Dictionary in Parameters :
            for Peak in Parameters[Dictionary] :
                for Parameter in Parameters[Dictionary][Peak] :
                    if Parameter in ParameterList :
                        for Key in Parameters[Dictionary][Peak][Parameter] :
                            if Key != 'set' :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+'='+str(Parameters[Dictionary][Peak][Parameter][Key]))
                            else :
                                exec('ModelParameters["'+Peak+'_'+Parameter+'"].'+Key+str(Parameters[Dictionary][Peak][Parameter][Key]))
        
        self.ModelParameters = ModelParameters
    
    def Fit(self,**kwargs) :
        
        for kwarg in kwargs :
            if kwarg == 'fit_x':
                fit_x = kwargs[kwarg]
        
        dt = DataTools()
        
        Data = self.Data
        Name = self.Name
        FitModel = self.FitModel
        ModelParameters = self.ModelParameters
        FitInfo = self.FitInfo
        
        if 'xRange' in FitInfo :
            Data = dt.TrimData(Data,FitInfo['xRange'][0],FitInfo['xRange'][1])
        x = Data.index.values
        try:
            fit_x
        except :
            try :
                NumberPoints
            except :
                fit_x = x
            else :
                for i in NumberPoints :
                    fit_x[i] = min(x) + i * (max(x) - min(x)) / (Numberpoints - 1)
        
        Fits = df(index=fit_x,columns=Data.columns.values)
        FitsParameters = df(index=ModelParameters.keys(),columns=Data.columns.values)
        FitsResults = list()
        FitsComponents = list()
        
        for idx,Column in enumerate(Data) :
            
            self.SetParameters(Column)
            
            y = Data[Column].values
            FitResults = FitModel.fit(y, ModelParameters, x=x, nan_policy='omit')
            fit_comps = FitResults.eval_components(FitResults.params, x=fit_x)
            fit_y = FitResults.eval(x=fit_x)
            ParameterNames = [i for i in FitResults.params.keys()]
            for Parameter in (ParameterNames) :
                FitsParameters[Column][Parameter] = FitResults.params[Parameter].value
            Fits[Column] = fit_y
            FitsResults.append(FitResults)
            FitsComponents.append(fit_comps)
            
            sys.stdout.write(("\rFitting %i out of "+str(Data.shape[1])) % (idx+1))
            sys.stdout.flush()
        
        self.Fits = Fits
        self.FitsParameters = FitsParameters
        self.FitsResults = FitsResults
        self.FitsComponents = FitsComponents
    
    def ShowFits(self,xLabel='',yLabel='') :
        
        Data = self.Data
        Fits = self.Fits
        FitInfo = self.FitInfo
        
        FitsParameters = self.FitsParameters
        FitsComponents = self.FitsComponents
        
        for idx,Column in enumerate(Data) :
            
            plt.figure(figsize = [6,4])
            plt.plot(Data.index, Data[Column],'k.', label='Data')
            plt.plot(Fits.index, Fits[Column], 'r-', label='Fit')
            for Component in FitsComponents[idx] :
                if not isinstance(FitsComponents[idx][Component],float) :
                    plt.fill(Fits.index, FitsComponents[idx][Component], '--', label=Component, alpha=0.5)
            plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(1.2, 1), ncol=1)
            plt.xlabel(xLabel), plt.ylabel(yLabel)
            if 'xRange' in FitInfo :
                plt.xlim(FitInfo['xRange'][0],FitInfo['xRange'][1])
            plt.title(str(Column))
            plt.show()
            
            Peaks = list()
            for Parameter in FitsParameters.index :
                Name = Parameter.split('_')[0]
                if Name not in Peaks :
                    Peaks.append(Name)

            string = ''
            for Peak in Peaks :
                string = string + Peak + ' | '
                for Parameter in FitsParameters.index :
                    if Peak == Parameter.split('_')[0] : 
                        string = string + Parameter.split('_')[1] + ': ' + str(round(FitsParameters[Column][Parameter],2))
                        string = string + ', '
                string = string[:-2] + '\n'
            print(string)
            print(75*'_')

##### TDS #####

class TDS :
    
    def __init__(self,DataFolder='') :
        
        self.dt = DataTools()
        self.Folders = Folders
        self.Folders['Data'] = DataFolder
    
    def LoadData(self, Folder, File) :
        
        dt = self.dt
        
        with open(Folder+'/'+File, 'r') as stream:
            Parameters = yaml.safe_load(stream)
        
        Parameters['FolderPath'] = self.Folders['Data']+'/'+Parameters['FolderPath']
        Data, Parameters = dt.Load_TDS(Parameters)
        if 'Assignments' in Parameters :
            Assignments = df(Parameters['Assignments'],index=Parameters['Masses'],columns=['Assignments'])
        else :
            Assignments = df(index=Parameters['Masses'],columns=['Assignments'])
        
        self.Assignments = Assignments
        self.ParametersFile = [Folder,File]
        self.Parameters = Parameters
        self.Data = Data
    
    def SimulateData(self,Rate) :
        
        Data = self.Data
        Parameters = self.Parameters
        
        if 'Simulations' in Parameters :

            # Initial parameters
            kB = 8.617e-5                 # eV/K
            T0 = 100                      # K
            
            Temperature, deltaT = np.linspace(min(Data.index),max(Data.index),1001,retstep =True)
            Time = Temperature / Rate
            deltat = deltaT / Rate
            Size = len(Temperature)
            
            Traces = df(index=Temperature)
            Coverages = df(index=Temperature)
            
            # Calculate traces
            for Mass in Parameters['Simulations'] :
                Trace = np.zeros((Size))
                Coverage = np.zeros((Size))
                for idx, Peak in enumerate(Parameters['Simulations'][Mass]) :
                    PeakParameters = Parameters['Simulations'][Mass][Peak]
                    Offset = PeakParameters['Offset']
                    Scaling = PeakParameters['Scaling']
                    Ni = PeakParameters['Coverage']
                    Ea = PeakParameters['Barrier']
                    nu = PeakParameters['Prefactor']
                    n = PeakParameters['Order']
                    
                    PeakTrace = np.zeros((Size))
                    PeakCoverage = np.zeros((Size))
                    IntRate = 0
                    for idx, T in enumerate(Temperature) :
                        PeakTrace[idx] = nu*(Ni - IntRate)**n * np.exp(-Ea/(kB*T))
                        IntRate += PeakTrace[idx] * deltat
                        PeakCoverage[idx] = Ni - IntRate
                        if IntRate >= Ni :
                            IntRate = Ni
                            PeakCoverage[idx] = 0
                        if PeakCoverage[idx] < 0 or PeakCoverage[idx] > Ni :
                            PeakCoverage[idx] = 0
                            PeakTrace[idx] = 0
                    Trace += PeakTrace * Scaling + Offset
                    Coverage += PeakCoverage
                
                Traces[Mass] = Trace
                Coverages[Mass] = Coverage
                
        self.SimulatedData = Traces
        self.SimulatedCoverages = Coverages
    
    def UI(self) :
        
        dt = self.dt
        out = widgets.Output()

        ##### Widgets #####

        self.ParametersFiles = widgets.Dropdown(
            options=dt.FileList(Folders['Parameters'],['.yaml','TPD']),
            description='Select File',
            layout=Layout(width='70%'),
            style = {'description_width': '150px'},
            disabled=False,
        )
        
        def Save2File_Clicked(b) :
            os.makedirs(Folders['Fits'], exist_ok=True)
            FitsFile = Folders['Fits'] + '/' + self.ParametersFiles.value + '.hdf'
            print(FitsFile)
            self.Data.to_hdf(FitsFile,'Data',mode='w')
            self.Assignments.to_hdf(FitsFile,'Assignments',mode='a')
        Save2File = widgets.Button(description="Save to File")
        Save2File.on_click(Save2File_Clicked)

        def SimulateTrace_Clicked(b) :
            with out :
                clear_output(True)

                self.LoadData(Folders['Parameters'],self.ParametersFiles.value+'.yaml')
                Data = self.Data
                self.SimulateData(self.HeatingRate.value)
                SimulatedData = self.SimulatedData

                fig = go.Figure()
                for Trace in Data :
                    if Trace != 'Time (s)' :
                        fig.add_trace(go.Scatter(x=Data.index,y=Data[Trace],name=Trace,mode='lines'))
                for Trace in SimulatedData :
                    fig.add_trace(go.Scatter(x=SimulatedData.index,y=SimulatedData[Trace],name=Trace,mode='lines'))
                fig.update_layout(xaxis_title='Temperature (K)',yaxis_title='Fit Value',title=self.Parameters['Description'],legend_title='')
                fig.show()

                fig = px.line(self.SimulatedCoverages)
                fig.update_layout(yaxis_title='Coverage',showlegend=False,height=100)
                fig.show()
                display(widgets.Box([SimulateTrace,self.HeatingRate]))
                display(Save2File)
        SimulateTrace = widgets.Button(description="Simulate Traces")
        SimulateTrace.on_click(SimulateTrace_Clicked)
        
        def ShowData_Clicked(b) :
            with out :
                clear_output(True)
                self.LoadData(Folders['Parameters'],self.ParametersFiles.value+'.yaml')
                Data = self.Data
                Masses = self.Parameters['Masses']
                fig = go.Figure()
                for Trace in Data :
                    if Trace != 'Time (s)' :
                        fig.add_trace(go.Scatter(x=Data.index,y=Data[Trace],name=Trace,mode='lines'))
                fig.update_layout(xaxis_title='Temperature (K)',yaxis_title='Fit Value',title=self.Parameters['Description'],legend_title='')
                fig.show()
                self.HeatingRate = widgets.FloatText(
                    value=np.around(self.Parameters['HeatingRate'],3),
                    description='Heating Rate (K/s):',
                    layout=Layout(width='25%'),
                    style = {'description_width': '140px'},
                    disabled=False
                )
                display(widgets.Box([SimulateTrace,self.HeatingRate]))
                display(widgets.HBox([Save2File]))
        ShowData = widgets.Button(description="Show Data")
        ShowData.on_click(ShowData_Clicked)
        
        display(self.ParametersFiles)
        display(ShowData)

        self.ParametersFile = self.ParametersFiles.value

        display(out)