from os import listdir
import numpy as np
from pandas import DataFrame as df
import struct
import yaml

def fileList(Filter) :
    
    FileList = [f for f in listdir()]
    for i in range(len(Filter)):
        FileList = [k for k in FileList if Filter[i] in k]
    for i in range(len(FileList)):
        FileList[i] = FileList[i].replace('.yaml','')
    
    return FileList
    
def loadData(Parameters) :
    
    FolderPath = Parameters['FolderPath']
    FileName = Parameters['FileName']

    if 'yaml' in Parameters['FileName'] or 'yml' in Parameters['FileName'] :

        with open(FolderPath + '/' + FileName, mode='rb') as file:
            fileContent = file.read()
        Data = yaml.safe_load(fileContent)
        del Data['Parameters']
        del Data['Pressure']

        Data = df.from_dict(Data)
        Data = Data.set_index('Temperature')
        
        Parameters['HeatingRate'] = np.mean(np.diff(Data.index)/np.diff(Data['Time']))

    else :
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
        Header.append('Time')
        if 'TChan' in Parameters :
            TChan = Parameters['TChan']
        else :
            TChan = 0
        for idx in range(NumChan) :
            if idx == TChan :
                Header.append('Temperature')
            else :
                Header.append(str(Masses[idx-1]))
        
        Data = df(np.transpose(Data),columns=Header)
        Data = Data.set_index('Temperature')
        if 'TScale' in Parameters :
            Data.index = Data.index * Parameters['TScale']
        
        Parameters['HeatingRate'] = np.mean(np.diff(Data.index)/np.diff(Data['Time']))
        
    return Data, Parameters

def removeEmptyDataSets(Data,Threshold) :
    
    Index = list()
    for i in Data.columns :
        if np.mean(Data[i]) < Threshold :
            Index.append(i)
    for i in Index :
        del Data[i]
    
    return Data

def trimData(Data,Min,Max) :
    
    Mask = np.all([Data.index.values>Min,Data.index.values<Max],axis=0)
    Data = Data[Mask]
    
    return Data

def reduceResolution(Data,Resolution=1) :
    
    Counter = 0
    ReducedData = df()
    for i in range(int(len(Data.columns.values)/Resolution)) :
        Column = round(np.mean(Data.columns[Counter:Counter+Resolution]),1)
        ReducedData[Column] = Data[Data.columns[Counter:Counter+Resolution]].mean(axis=1)
        Counter = Counter + Resolution
    
    return ReducedData