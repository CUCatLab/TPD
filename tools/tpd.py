import os
from os import listdir
import sys
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import yaml
import ipywidgets as ipw
from ipywidgets import Button, Layout
from IPython.display import clear_output
import matplotlib.pyplot as plt
import cmath
import re
import yaml

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
    legend_x = 1,
    legend_y = 1,
    legend_traceorder = "normal",
    legend_bgcolor='rgba(0,0,0,0)'
))

parametersFile = 'tools/parameters.yaml'

class dataTools :
    
    def __init__(self) :

        pass

    def loadData(self, file, folder) :

        def loadTPD_SU(parameters) :
            
            FolderPath = parameters['FolderPath']
            fileName = parameters['FileName']
            Masses = parameters['Masses']

            with open(FolderPath + '/' + fileName, mode='rb') as file:
                fileContent = file.read()

            NumChan = len(Masses) + 1
            dataLength = int((len(fileContent)-5)/(46*NumChan))
            data = np.zeros((int(1+NumChan),dataLength))

            for i in range(len (data)) :
                for j in range(len (data[0])) :
                    if i == 0 :
                        index = int(31+j*46*NumChan)
                        data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]/1000
                    else :
                        index = int(43+j*46*NumChan + (i-1)*46)
                        data[i,j] = struct.unpack('<d', fileContent[index:index+8])[0]
            
            Header = list()
            Header.append('Time (s)')
            if 'TChan' in parameters :
                TChan = parameters['TChan']
            else :
                TChan = 0
            for idx in range(NumChan) :
                if idx == TChan :
                    Header.append('Temperature (K)')
                else :
                    Header.append(str(Masses[idx-1]))
            
            data = df(np.transpose (data),columns=Header)
            if 'Temperature (K)' in data:
                data = data.set_index('Temperature (K)')
            else:
                data = data.set_index('Temperature')
            if 'TScale' in parameters :
                data.index = data.index * parameters['TScale']
            
            parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time (s)']))
                
            return data, parameters

        def loadTPD(parameters) :
            
            FolderPath = parameters['FolderPath']
            fileName = parameters['FileName']
            
            if 'yml' in fileName:
                with open(FolderPath + '/' + fileName, mode='rb') as file:
                    fileContent = file.read()
                allData = yaml.safe_load(fileContent)
                
                data = dict()
                
                for key in allData:
                    try:
                        data[float(key)] = allData[key]
                    except ValueError:
                        data[key] = allData[key]
                        
            else:
                with open(FolderPath + '/' + fileName, mode='rb') as file:
                    fileContent = pd.read_csv(file)

                data = fileContent
            
            if 'Parameters' in data:
                del data['Parameters']
            if 'Pressure (Torr)' in data:
                del data['Pressure (Torr)']
            if 'Pressure' in data:
                del data['Pressure']
            if 'Mass 15.0' in data:
                data['CH3'] = data['Mass 15.0']
                del data['Mass 15.0']
            if 'Mass 17.0' in data:
                data['OH'] = data['Mass 17.0']
                del data['Mass 17.0']
            if 'Mass 18.0' in data:
                data['H2O'] = data['Mass 18.0']
                del data['Mass 18.0']
            if 'Mass 2.0' in data:
                data['H2'] = data['Mass 2.0']
                del data['Mass 2.0']
            if 'Mass 28.0' in data:
                data['CO'] = data['Mass 28.0']
                del data['Mass 28.0']
            if 'Mass 32.0' in data:
                data['O2'] = data['Mass 32.0']
                del data['Mass 32.0']
            if 'Mass 44.0' in data:
                data['CO2'] = data['Mass 44.0']
                del data['Mass 44.0']
            
            if type (data) == dict:
                data = df.from_dict (data)
            if 'Temperature (K)' in data:
                data = data.set_index('Temperature (K)')
            else:
                data = data.set_index('Temperature')
            
            if 'Time (s)' in data:
                parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time (s)']))
            else:
                parameters['HeatingRate'] = np.mean(np.diff (data.index)/np.diff (data['Time']))
                
            return data, parameters

        with open(file, 'r') as stream :
            parameters = yaml.safe_load(stream)
        
        if 'FolderPath' not in parameters :
            print(file)
            if 'TPD' in file or 'tpd' in file:
                date = re.split(r'TPD|_',file)[1]
            if 'tpd' in file :
                date = re.split(r'tpd|_',file)[1]
            if len(date) == 6 :
                date = '20'+date
            print(date[2:4]+'.'+date[4:6]+'.'+date[6:8])
            parameters['FolderPath'] = folder+'/'+'20'+date[2:4]+'/'+'20'+date[2:4]+'.'+date[4:6]+'.'+date[6:8]
        
        if 'fileName' not in parameters :
            # print(file)
            parameters['fileName'] = file

        try :
            data, parameters = loadTPD(parameters)
        except :
            data, parameters = loadTPD_SU(parameters)
        
        listy = []
        for i in data.keys():
            listy.append(i)
        
        return data, parameters
    
    def plotData(self,data) :

        fig = go.Figure()
        for Trace in data :
            if Trace != 'Time (s)' and Trace != 'Time':
                fig.add_trace(go.Scatter(x=data.index,y=data[Trace],name=Trace,mode='lines'))
        fig.update_layout(xaxis_title='Temperature (K)',yaxis_title='Fit Value',legend_title='')
        fig.show()


class analysisTools :
    
    def __init__(self) :

        pass

    def simulateData(self, data, parameters, Rate) :
        
        if 'Simulations' in parameters :

            # Initial parameters
            kB = 8.617e-5                 # eV/K
            T0 = 100                      # K
            
            Temperature, deltaT = np.linspace(min (data.index),max (data.index),1001,retstep =True)
            Time = Temperature / Rate
            deltat = deltaT / Rate
            Size = len(Temperature)
            
            Traces = df(index=Temperature)
            Coverages = df(index=Temperature)
            
            # Calculate traces
            for Mass in parameters['Simulations'] :
                Trace = np.zeros((Size))
                Coverage = np.zeros((Size))
                for idx, Peak in enumerate(parameters['Simulations'][Mass]) :
                    PeakParameters = parameters['Simulations'][Mass][Peak]
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
                
        return Traces, Coverages


class UI :
    
    def __init__(self) :

        dt = dataTools()
        at = analysisTools()

        self.cwd = Path(os.getcwd())

        self.FoldersLabel = '-------Folders-------'
        self.FilesLabel = '-------Files-------'
        self.parametersFile = parametersFile
        
        with open(parametersFile, 'r') as stream :
            self.folders = yaml.safe_load(stream)['folders']
        
        out = ipw.Output()

        dataFolder = ipw.Text(value=self.folders['data'],
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Data Folder')

        def changeDataFolder(value) :
            if value['new'] :
                with open(self.parametersFile, 'r') as f :
                    data = yaml.safe_load(f)
                data['folders']['data'] = dataFolder.value
                self.folders['data'] = dataFolder.value
                with open(self.parametersFile, 'w') as f:
                    yaml.dump(data, f)
        dataFolder.observe(changeDataFolder, names='value')

        def go_to_address(address):
            address = Path(address)
            if address.is_dir():
                currentFolder_field.value = str(address)
                SelectFolder.unobserve(selecting, names='value')
                SelectFolder.options = self.get_folder_contents(folder=address)[0]
                SelectFolder.observe(selecting, names='value')
                SelectFolder.value = None
                selectFile.options = self.get_folder_contents(folder=address)[1]

        def newaddress(value):
            go_to_address(currentFolder_field.value)
        currentFolder_field = ipw.Text(value=str(self.cwd),
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Current Folder')
        currentFolder_field.on_submit(newaddress)
                
        def selecting(value) :
            if value['new'] and value['new'] not in [self.FoldersLabel, self.FilesLabel] :
                path = Path(currentFolder_field.value)
                newpath = path / value['new']
                if newpath.is_dir():
                    go_to_address(newpath)
                elif newpath.is_file():
                    #some other condition
                    pass
        
        SelectFolder = ipw.Select(
            options=self.get_folder_contents(self.cwd)[0],
            rows=5,
            value=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Subfolders')
        SelectFolder.observe(selecting, names='value')
        
        selectFile = ipw.Select(
            options=self.get_folder_contents(self.cwd)[1],
            rows=10,
            values=None,
            layout=Layout(width='70%'),
            style = {'width': '100px','description_width': '150px'},
            description='Files')

        def parent(value):
            new = Path(currentFolder_field.value).parent
            go_to_address(new)
        up_button = ipw.Button(description='Up',layout=Layout(width='10%'))
        up_button.on_click(parent)

        def ShowData_Clicked(b) :
            with out :
                clear_output(True)
                data, parameters = dt.loadData(selectFile.value,dataFolder.value)
                self.data = data
                self.parameters = parameters
                dt.plotData(data)
        ShowData = ipw.Button(description="Show data")
        ShowData.on_click(ShowData_Clicked)

        def SimulateTrace_Clicked(b) :
            with out :
                clear_output(True)
                data, parameters = dt.loadData(selectFile.value,dataFolder.value)
                self.data = data
                self.parameters = parameters
                HeatingRate = ipw.FloatText(
                    value=np.around(self.parameters['HeatingRate'],3),
                    description='Heating Rate (K/s):',
                    layout=Layout(width='25%'),
                    style = {'description_width': '140px'},
                    disabled=False
                    )
                simulatedData, SimulatedCoverages = at.simulateData(data, parameters, HeatingRate.value)
                self.simulatedData = simulatedData
                self.SimulatedCoverages = SimulatedCoverages
                allData = pd.concat([data,simulatedData])
                dt.plotData(allData)
                fig = px.line(SimulatedCoverages)
                fig.update_layout(yaxis_title='Coverage',showlegend=False,height=400)
                fig.show()
                display(ipw.Box([SimulateTrace,HeatingRate]))
        SimulateTrace = ipw.Button(description="Simulate Traces")
        SimulateTrace.on_click(SimulateTrace_Clicked)
        
        display(ipw.HBox([dataFolder]))
        display(ipw.HBox([currentFolder_field]))
        display(ipw.HBox([SelectFolder,up_button]))
        display(ipw.HBox([selectFile]))
        display(ipw.HBox([ShowData,SimulateTrace]))

        self.parametersFile = selectFile.value

        display(out)

    def get_folder_contents(self,folder):

        'Gets contents of folder, sorting by folder then files, hiding hidden things'
        folder = Path(folder)
        folders = [item.name for item in folder.iterdir() if item.is_dir() and not item.name.startswith('.')]
        files = [item.name for item in folder.iterdir() if item.is_file() and not item.name.startswith('.')]
        return sorted(folders), sorted(files)
