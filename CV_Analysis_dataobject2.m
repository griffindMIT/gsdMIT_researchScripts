function CV_Analysis_dataobject2()

%Needs support for - being called from external functions
%Theoretical inputs - manual (toggle, 1 if manual, 0 if automatic)
%dateVector = [yyyymmdd,yyyymmdd,yyyymmdd,...] vector of input dates. 
%experimentpMat = {[1,10,13,51],'all',[15,20],...} cell containing either a
%list of experiments or keyword 'all'. dateVector{i} ~ experimentpMat{i}

%Ask user inputs, load corresponding files - easy peasy
%The file in question? an Excel file (DATE_workup)
%Each sheet is 1 experiment. Two columns, V & I, w/ all traces.
%Sheet name is "X", corresponding to NX on DATE. 
%"Scan" sheet (end-1) contains:
%A1-A#: experiment #.
%B1-B#: scan rate corresp. to each sheet in rest of xcel
%C1-C#: conc. of substrate (mM) corresp. to each sheet in rest of xcel
%"Info" sheet (end) contains:
%A1: Information
%A2: DATE (yyyymmdd)
%A3: Surface area (cm^2)
%A4: Compound name (bibenzyl)

%Set path, set importing & indexing variables, preallocat matrix...
dataPath = 'C:\Users\ChemeGrad2019\Dropbox (MIT)\Griffin\Data';
file_importing = 0;
dataMat = {};
dMS = 1;

%While file importing is still active (0)
while file_importing == 0
    %Ask for input (add file, done)
    response = input('Input file command <add>/<done>: ','s');
    response = lower(response);
    %Add a file?
    if strcmp(response,'add')
        %Select a date
        date = input('Input date (yyyymmdd): ','s');
        try
            %Try to load the list of sheets...
            handle = [dataPath,'\',date,'_CVs\','workup2_',date,'.xlsx'];
            [~,sheets] = xlsfinfo(handle);
            %Get experiment-wide information (date, species)
            infoData = readtable(handle,'Sheet',sheets{end});
            infoStruct.date = str2double(infoData{1,1}{1});
            %infoStruct.SA = str2double(infoData{2,1}{1}); - unused, couldb
            infoStruct.species = infoData{3,1}{1};
            %infoStruct.conc = str2double(infoData{4,1}{1}); - unus.,couldb
            %Remove info sheet from list (not needed)
            sheets(end) = [];
            extraDat = readtable(handle,'Sheet',sheets{end});
            numsheets = extraDat{:,1};
            scanRates = extraDat{:,2};
            concs = extraDat{:,3};
            %Remove scan sheet
            sheets(end) = [];
            %Alert user no problems
            fprintf('Day successfully loaded. \n');
            exp_importing = 0;
        catch
            %If something broke, the excel is probably busted and bounces
            %the user. 
           fprintf('Bad date/experiment. Returning to input. \n');
           exp_importing = 1;
        end
        %This begins the phase of 'experiment importing.'
        while exp_importing == 0
           %Ask the user the input - add 1, add all, or done.
           response2 = input('Input experiment command <add>/<all>/<done>: ','s');
           response2 = lower(response2);
           %Add 1? 
           if strcmp(response2,'add')
               %Input the #...
               expNo = input('Import experiment #: ');
               %Try this # (might not be OK, numerical input)
               try
                   %Gets I/V data - reads the excel file
                   data = readtable(handle,'Sheet',sheets{numsheets == expNo});
                   data = data{:,:};
                   %Grabs the current scan rate
                   scan = scanRates(numsheets == expNo);
                   conc = concs(numsheets == expNo);
                   %Stores everything relevant in dataMat columns
                   dataMat{dMS,1} = data(:,1); %V
                   dataMat{dMS,2} = data(:,2); %I
                   dataMat{dMS,3} = scan/1000; %scan rate, V/s
                   dataMat{dMS,4} = conc; %conc, mM
                   dataMat{dMS,5} = expNo; %Experiment #
                   dataMat{dMS,6} = infoStruct.date; %Date run
                   dataMat{dMS,7} = infoStruct.species; %Species
                   dataMat{dMS,8} = dMS; %Counter
                   %Increments dataMatSize by 1
                   dMS = dMS + 1;
               catch
                  %Alert the user to issues
                  fprintf('Experiment # failed. \n'); 
               end
           elseif strcmp(response2,'all')
               %Alert the user to choice...
               fprintf('Importing all experiments. \n');
               for i = 1:length(sheets)
                  %Basically repeat "add 1" procedure for all experiments
                  failure = 0;
                  expNo = numsheets(i);
                  try
                    %Gets I/V data
                    data = readtable(handle,'Sheet',sheets{numsheets == expNo});
                    data = data{:,:};
                    scan = scanRates(numsheets == expNo);
                    conc = concs(numsheets == expNo);
                    %Stores everything in dataMat
                    dataMat{dMS,1} = data(:,1); %V
                    dataMat{dMS,2} = data(:,2); %I
                    dataMat{dMS,3} = scan/1000; %scan rate, V/s
                    dataMat{dMS,4} = conc; %conc, mM
                    dataMat{dMS,5} = expNo; %Experiment #
                    dataMat{dMS,6} = infoStruct.date; %Date run
                    dataMat{dMS,7} = infoStruct.species; %Species
                    dataMat{dMS,8} = dMS; %Counter
                    dMS = dMS + 1;
                  catch
                    %Log if any experiments failed to load  
                    failure = 1;
                    fprintf('Experiment #%i failed. \n',expNo); 
                  end
               end
               if failure == 0
                   %If everything went smoothly the file is deffo done. 
                   exp_importing = 1;
                   fprintf('Done importing all experiments from date. \n');
               else
                   %Otherwise the user might want to debug or w/e. 
                   fprintf('Not all experiments were imported properly. \n');
               end
           elseif strcmp(response2,'done')
               %Done adding experiments? exit loop. 
               fprintf('Exiting experiment addition loop. \n');
               exp_importing = 1;
           else 
              %Failsafe for bad inputs. 
              fprintf('Response not understood. \n');
           end
        end
    elseif strcmp(response,'done')
        %Done adding files? exit loop. 
        fprintf('Exiting file addition loop. \n');
        file_importing = 1;
    else
        %Failsafe for bad inputs. 
        fprintf('Response not understood. \n');
    end
end

response3 = input('Update dataMat externally? y/n ','s');
    %This overwrites whatever dataMat might previously have existed.
    if strcmp(response3,'y')
        %yes: overwrite. 
        fprintf('Writing new dataMat. \n');
        assignin('base','dataMat',dataMat);
    else 
        %no: don't do it. pessimistic view here. 
        fprintf('New dataMat not written. \n');
    end
end

