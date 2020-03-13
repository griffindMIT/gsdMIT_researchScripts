function [dataMat,infoStruct] = CV_Analysis_dataobject(dataDay,dataFile)
close all; clc;

%Inputs: 
    %dataDay: name of folder
    %dataFile: name of workup file
    %Allowable to make these as an input for collating functions later.

%Outputs:
    %DataStruct is a 2d cell array containing: 
    %Column 1: Voltages for the final (3rd) CV trace, in V vs Ag/AgCl
    %Column 2: Currents for the final (3rd) CV trace, in A
    %Column 3: Scan rates for the CV, in V/s
    %Column 4: Peak #, onset/peak/outset??? - not supported yet!! -
    %This information is stored as column1/column2/sheettitle in Excel (mV/s)

    %InfoStruct is a struct containing: 
    %infoStruct.date: the date the experiment was run (A2)
    %infoStruct.SA: the geometric surface area of the working electrode, cm^2 (A3)
    %infoStruct.species: the species in use (A4)
    %infoStruct.conc: the concentration of the above species, mM (A5)
    %This information is stored in the 'info' sheet in the workup excel on the
    %specified series_workup. Info should be the last sheet.

%Permanent data pathway...
dataPath = 'C:\Users\ChemeGrad2019\Dropbox (MIT)\Griffin\Data';

%Create handle, grab list of sheets. Info should be last sheet. 
handle = [dataPath,'\',dataDay,'\',dataFile];
[~,sheets] = xlsfinfo(handle);
infoData = readtable(handle,'Sheet',sheets{end});

%[numdat,textdat] = xlsread(handle,sheets{end});
infoStruct.date = str2double(infoData{1,1}{1});
infoStruct.SA = str2double(infoData{2,1}{1});
infoStruct.species = infoData{3,1}{1};
infoStruct.conc = str2double(infoData{4,1}{1});

%Remove info sheet for further processing;
sheets(end) = [];
experCt = length(sheets);
dataMat{experCt,3} = [];

for exper = 1:length(sheets)
    %Reads as table and converts into matrix
    data = readtable(handle,'Sheet',sheets{exper});
    data = data{:,:};
    dataMat{exper,1} = data(:,1); %voltages
    dataMat{exper,2} = data(:,2); %currents
    dataMat{exper,3} = str2double(sheets{exper})/1000; %mV/s -> V/s
end

end

