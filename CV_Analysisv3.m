function CV_Analysisv3()
close all; clc;
%Uses finite difference derivative/second derivative methods. 

%Set path/file of interest...
dataPath = 'C:\Users\ChemeGrad2019\Dropbox (MIT)\Griffin\Data';
dataDay = '20200219_CVs';
dataFile = 'workup_20200219.xlsx';
%Get experiment date
expDate1 = str2double(dataDay(1:8));
expDate2 = expDate1;
peakHandle = [dataPath,'\',dataDay,'\',dataDay(1:8),'_peaks'];
commentHandle = [dataPath,'\',dataDay,'\',dataDay(1:8),'_comments'];
nameHandle = [dataPath,'\',dataDay,'\',dataDay(1:8),'_notebook'];
scanHandle = [dataPath,'\',dataDay,'\',dataDay(1:8),'_scans'];

%%%%%%%%%%%%%%%%%%%%% TOGGLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
oxidationOnset = 1; %V 
FTN1 = 25; %Filtering #
FTN2 = 25;
blankScanRate = 98.631/1000; %mV/s -> V/s, scan rate for blank runs. 
write = 0; %Write peak ID to text file & displays normalized CVs.
read = 1; %Creates plots at the end.
displayCVs = 1; %Simply displays CVs

%%%%%%%%%%%%%%%%%%%% IMPORT VOLTAGES/CURRENTS/INFO %%%%%%%%%%%%%%%%%%%%%

%A few headache-saving checks...
if write == 1 && read == 1
    error('Write and read mutually exclusive.');
elseif read == 1
    try 
        fID = fopen([peakHandle,'.txt']);
        fclose(fID);
    catch
       error('File does not exist yet. Switch to write mode.'); 
    end
end

%If the file doesn't exist yet... 
if write == 1
    try
        fID = fopen([peakHandle,'.txt']);
        fclose(fID);
    catch
        fID = fopen([peakHandle,'.txt'],'w');
        t = table(0,'VariableNames',{'peaks'});
        writetable(t,[peakHandle,'.txt']);
        fclose(fID);
    end
    try 
        fID = fopen([commentHandle,'.txt']);
        fclose(fID);
    catch
        fID = fopen([commentHandle,'.txt'],'w');
        t = table({'hello';'there'},'VariableNames',{'comments'});
        writetable(t,[commentHandle,'.txt']);
        fclose(fID);
    end
    try 
        fID = fopen([nameHandle,'.txt']);
        fclose(fID);
    catch
        fID = fopen([nameHandle,'.txt'],'w');
        t = table({'hello';'there'},'VariableNames',{'notebook'});
        writetable(t,[nameHandle,'.txt']);
        fclose(fID);
    end
end
        

%Check if dataMat already exists
dataMatExists = evalin('base','exist(''dataMat'',''var'') == 1');
if dataMatExists == 1
    %Import the datamat & struct information if it does.
    dataMat = evalin('base','dataMat');
    infoStruct = evalin('base','infoStruct');
    expDate2 = infoStruct.date;
    %Note that changes to dataMat are /not/ stored unless write fcn is
    %called.
end
%If the dataMat does not exist or the experiment date does not match...
if dataMatExists == 0 || expDate1 ~= expDate2
    %Generate a new dataMat & populate it in the main workspace.
    [dataMat,infoStruct] = CV_Analysis_dataobject(dataDay,dataFile);
    assignin('base','dataMat',dataMat);
    assignin('base','infoStruct',infoStruct);
end

%%%%%%%%%%%%%%%%%%%%%%%%% VOLTAGE/CURRENT DATA PROCESSING %%%%%%%%%%%%%%%
%Determine # of experiments
NoExps = length(dataMat(:,1));
derivPeakMat = zeros(NoExps,1);
existingPeaks = [];
peakCt = 1;
peakTypes = {'ro','kx','bd','gv','m>','c<','yp'};
lineTypes = {'r','k','b','g','m','c','y'};
review = 0;
closeMode = 0;

%For each experiment...
for b = 1:NoExps
    %Mark the positions where the trace is at the open circuit potential
    OCPs = find(dataMat{b,1}==dataMat{b,1}(end));
    if dataMat{b,1}(end) > oxidationOnset
        OCPs = [1,0,length(dataMat{b,1})]; %The Gamry Correction
    end
    %Clip the voltages and currents to the last trace (two OCPs ago).
    LastTrace{b,1} = dataMat{b,1}(OCPs(end-2):end);
    LastTrace{b,2} = dataMat{b,2}(OCPs(end-2):end);
    %dataMat{b,1} = dataMat{b,1}(OCPs(end-2):end);
    %dataMat{b,2} = dataMat{b,2}(OCPs(end-2):end);
    %Plot the CV trace in the relevant area
    if displayCVs == 1
        CVplot(b,NoExps,dataMat,0);
    end
    %For analysis, get only the oxidative trace
    %Only values past the onset of oxidation...
    %oxidV = dataMat{b,1}(dataMat{b,1}>oxidationOnset);
    %oxidI = dataMat{b,2}(dataMat{b,1}>oxidationOnset);
    oxidV = LastTrace{b,1}(LastTrace{b,1}>oxidationOnset);
    oxidI = LastTrace{b,2}(LastTrace{b,1}>oxidationOnset);
    %Find turnaround index
    if dataMat{b,1}(end) < oxidationOnset
    returnIndex = find(oxidV == max(oxidV));
    oxidV = oxidV(1:returnIndex);
    oxidI = oxidI(1:returnIndex);
    else
        %The Gamry Correction
        [~,returnIndex] = min(oxidV);
        oxidV = oxidV(returnIndex:end);
        oxidI = oxidI(returnIndex:end);
        %oxidV = oxidV(1:returnIndex);
        %oxidI = oxidI(1:returnIndex);
    end
    %Compute first derivative with FD coeffs (lose 2 points per end).
    Ideriv = (oxidI(1:end-4)*(1/12) - oxidI(2:end-3)*(2/3) ...
         + oxidI(4:end-1)*(2/3) - oxidI(5:end)*(1/12));
    IderivFT = smooth(oxidV(3:end-2),Ideriv,FTN1);
    IderivFT = IderivFT/max(IderivFT);
    %Compute second deriv with FD coeffs (lose 2 points per end).
    I2deriv = IderivFT(1:end-4)*(1/12) - IderivFT(2:end-3)*(2/3) ...
     + IderivFT(4:end-1)*(2/3) - IderivFT(5:end)*(1/12);
    I2derivFT = smooth(oxidV(5:end-4),I2deriv,FTN2);
    I2derivFT = I2derivFT/max(I2derivFT);
    %Force congruency between datasets
    oxidV = oxidV(5:end-4);
    IderivFT = IderivFT(3:end-2);
    %Generate useful figure for the user to identify peaks
    figure(b); hold on;
    xlabel('Potential vs. Ag/AgCl in ACN (V)');
    ylabel('Normalized (to max) Current Deriv. (-)');
    title(sprintf('Scan rate: %0.4f V/s',dataMat{b,3}));
    set(gca,'FontSize',14)
    plot(oxidV,IderivFT,'-k','LineWidth',1,'DisplayName','Deriv');
    ylim([-0.1,2.5]);
    yyaxis right
    ylabel('Normalized (to max) Current 2nd Deriv. (-)');
    ylim([-2.5,1]);
    set(gca,'ycolor','b');
    plot(oxidV,I2derivFT,'-b','LineWidth',1,'DisplayName','2nd Deriv');
    plot([oxidV(1),oxidV(end)],[0,0],'--r','LineWidth',1,'DisplayName','2nd Deriv Zero');
    %Update existing tables
    
    
    %Begin bifurcation...
    if write == 1
        %Write mode: select peaks and write their locations to a text file
        doneWriting = 0;
        fprintf('Write mode. <> Indicates commands. Current scan: %0.4f V/s. \n',dataMat{b,3});
        fprintf('Type <help> for a list of commands. \n');
        cursor = imdistline(gca,[2,2],[-10,10]);
        setLabelVisible(cursor,0);
        while doneWriting == 0
            response = input('Input command: ','s');
            response = lower(response);
            if strcmp(response,'exit')
                fprintf('Exiting write mode.\n');
                doneWriting = 1;
                write = 0;
            elseif strcmp(response,'help')
                fprintf('<exit> - exits write mode. All derivative plots still generated. \n');
                fprintf('<next> - go to next scan. You will be asked if you want to save peak data. \n');
                fprintf('<peak> - place peak. You will be asked to confirm the peak ID/#. \n');
                fprintf('<up#>/<down#> - vertical zoom in/out on #th derivative (1 or 2). \n');
                fprintf('<xin>/<xout> - zooms in x axis at cursor location/resets zoom. \n')
                fprintf('<scomm>/<ecomm> - see/edit analysis comments. \n');
                fprintf('<sname>/<ename> - see/edit experiment name. \n');
                fprintf('<getpos> - gets current position of cursor (voltage). \n');
                fprintf('<review> - enters/exits review mode (no prompt to save data). \n');
                fprintf('<closing> - enters/exits closing mode (only current figures visible). \n');
                fprintf('<cvshow>/<cvmark> - shows corresponding cv/plots vertical line at spec voltage. \n');
            elseif strcmp(response,'scomm') || strcmp(response,'ecomm')
                tt1 = readtable([commentHandle,'.txt']);
                tt1 = tt1{:,1};                
                try
                  fprintf(['Comment: ',tt1{b,1},'\n']);
                catch
                  fprintf('No comment exists.\n');
                end
                if strcmp(response,'ecomm')
                    newcomm = input('Enter new comment or <undo>: ','s');
                    if strcmp(newcomm,'undo')
                       fprintf('No new comment saved. \n'); 
                    else
                       for kk = 1:b
                          if isempty(tt1{kk,1})
                             tt1{kk,1} = 'blank'; 
                          end
                       end 
                       tt1{b,1} = {newcomm};
                       tt1 = table(tt1,'VariableNames',{'comments'});
                       writetable(tt1,[commentHandle,'.txt']);
                       fprintf('Comment added. \n');
                    end
                end
            elseif strcmp(response,'sname') || strcmp(response,'ename')
                tt1 = readtable([nameHandle,'.txt']);
                tt1 = tt1{:,1};
                try
                  fprintf(['Name: ',tt1{b,1},'\n']);
                catch
                  fprintf('No name exists.\n');
                end
                if strcmp(response,'ename')
                    newcomm = input('Enter new name or <undo>: ','s');
                    if strcmp(newcomm,'undo')
                       fprintf('No new name saved. \n'); 
                    else
                       tt1{b,1} = newcomm;
                       for kk = 1:b
                          if isempty(tt1{kk,1})
                             tt1{kk,1} = 'blank'; 
                          end
                       end
                       tt1 = table(tt1,'VariableNames',{'names'});
                       writetable(tt1,[nameHandle,'.txt']);
                       fprintf('Name added. \n');
                    end
                end
            elseif strcmp(response,'review')
                if review == 0
                   fprintf('Entering review mode. \n');
                   review = 1;
                else
                   fprintf('Exiting review mode. \n');
                   review = 0;
                end
            elseif strcmp(response,'closing')
                if closeMode == 0
                    fprintf('Close mode activated. \n');
                    closeMode = 1;
                else
                    fprintf('Close mode disabled. \n');
                    closeMode = 0;
                end
            elseif strcmp(response,'getpos')
                peakVolt = getPosition(cursor);
                peakVolt = peakVolt(1,1);
                fprintf('Cursor at at %0.10f. V\n',peakVolt);
            elseif strcmp(response,'cvshow') || strcmp(response,'cvmark')
                str = sprintf('CV has %0.f traces. Select trace # (0 for all): ',floor(length([1;OCPs])/2));
                tN = input(str);
                CVplot(b,NoExps,dataMat,tN);
                if strcmp(response,'cvmark')
                   figure(b+NoExps); hold on
                   try
                    spec = input('Markup at voltage: ');
                    plot([spec, spec],[-10,10],'--r');
                   catch
                    fprintf('Error. Please input numerical voltage value.');
                   end
                end
            elseif strcmp(response,'up1') || strcmp(response,'up2') || strcmp(response,'down1') || strcmp(response,'down2')
                if strcmp(response,'up1') || strcmp(response,'down1')
                    yyaxis left
                    limits = ylim;
                    if strcmp(response,'up1')
                       limits = limits*0.5;
                    else
                       limits = limits*2;
                    end
                else
                    yyaxis right
                    limits = ylim;
                    if strcmp(response,'up2')
                       limits = limits*0.5;
                    else
                       limits = limits*2;
                    end
                end 
                ylim(limits)
            elseif strcmp(response,'xin') || strcmp(response,'xout')
                if strcmp(response,'xin')
                    peakVolt = getPosition(cursor);
                    peakVolt = peakVolt(1,1);
                    xlim([peakVolt-0.25,peakVolt+0.25]);
                else
                    xlim([oxidationOnset,max(dataMat{b,1})]);
                end
            elseif strcmp(response,'next')
                fprintf('Moving to next scan. \n');
                if review == 0
                    repso = input('Update text file? Overwrites previous scan data! <yes>/<no>: ','s');
                else
                    repso = 'no';
                end
                if strcmp(repso,'yes')
                    tt1 = readtable([peakHandle,'.txt']);
                    derivPeakMatFile = tt1{:,:};
                    for jj = 1:length(derivPeakMat(b,:))
                        derivPeakMatFile(b,jj) = derivPeakMat(b,jj);
                    end
                        tt2 = table(derivPeakMatFile,'VariableNames',{'peaks'});
                        writetable(tt2,[peakHandle,'.txt']);
                        fprintf('File updated. \n');
                else
                        fprintf('File not updated. \n');
                end
                doneWriting = 1;
            elseif strcmp(response,'peak')
                peakVolt = getPosition(cursor);
                peakVolt = peakVolt(1,1);
                fprintf('Peak input at %0.10f V.\n',peakVolt);
                str = '';
                for j = 1:length(existingPeaks)
                    if j < length(existingPeaks)
                        str = [str,num2str(existingPeaks(j)),', '];
                    else
                        str = [str,num2str(existingPeaks(j))];
                    end
                end
                fprintf(['Existing Peak ID#s: ',str,' \n']);
                newNo = input('Assign peak ID#: ');
                if sum(ismember(existingPeaks,newNo)) == 0
                    existingPeaks(peakCt) = newNo;
                    peakCt = peakCt + 1;
                end
                derivPeakMat(b,newNo) = peakVolt;
                yyaxis left
                text(peakVolt,0.5,['#',num2str(newNo)],'Color','r');
                plot([peakVolt,peakVolt],[-5,5],'--k','LineWidth',0.5);
                addAgain = input('Back to input mode for current scan? <yes>/<no>: ','s');
                if strcmp(addAgain,'no')
                    if review == 0
                        repso = input('Update text file? Overwrites previous scan data! <yes>/<no>: ','s');
                    else
                        repso = 'no'; 
                    end
                    if strcmp(repso,'yes')
                        tt1 = readtable([peakHandle,'.txt']);
                        derivPeakMatFile = tt1{:,:};
                        for jj = 1:length(derivPeakMat(b,:))
                            derivPeakMatFile(b,jj) = derivPeakMat(b,jj);
                        end
                        if jj < length(derivPeakMatFile(b,:))
                           derivPeakMatFile(b,(jj+1):end) = 0; 
                        end
                        tt2 = table(derivPeakMatFile,'VariableNames',{'peaks'});
                        writetable(tt2,[peakHandle,'.txt']);
                        fprintf('File updated. \n');
                    else
                        fprintf('File not updated. \n');
                    end
                    doneWriting = 1;
                else
                   fprintf('Returning to input mode. \n'); 
                end
            else
                fprintf('Improper command. Try <help> for command list. \n');
            end
            figure(b);
        end
    end
    if closeMode == 1
       figure(b); close
       figure(b+NoExps); close
    end
end

if read == 1
    %Read mode: open text file and extract relevant peak data
    fprintf('Entering read mode. \n');
    %Open the text file as a table
    tt1 = readtable([peakHandle,'.txt']);
    derivPeakMat = tt1{:,:};
    scanRates = zeros(1,NoExps);
    for b = 1:NoExps
       scanRates(b) = dataMat{b,3}; 
    end
    scanTable = table(scanRates','VariableNames',{'scans'});
    writetable(scanTable,scanHandle);
    transScanRates = log10(scanRates)';
    for p = 1%:length(derivPeakMat(1,:))
       for b = 1:NoExps
          if derivPeakMat(b,p) ~= 0 
              figure(b); hold on;
              yyaxis left 
              text(derivPeakMat(b,p),0.5,['#',num2str(p)],'Color','r');
              plot([derivPeakMat(b,p),derivPeakMat(b,p)],[-5,5],'--k','LineWidth',0.5);
          end
       end
       figure(2*NoExps + 1); hold on;
       relevantPeaks = derivPeakMat(:,p);
       relevantScans = transScanRates(relevantPeaks>0);
       relevantPeaks = relevantPeaks(relevantPeaks>0);
       %Removal of blanks here...
       relevantPeaks = relevantPeaks(relevantScans ~= log10(blankScanRate));
       relevantScans = relevantScans(relevantScans ~= log10(blankScanRate));
       if length(relevantPeaks) > 1 
        coeffs = polyfit(relevantScans,relevantPeaks,1);
        regline = polyval(coeffs,relevantScans);
        %plot(relevantScans,relevantPeaks,peakTypes{p},'DisplayName',['Peak # ',num2str(p)]);
        plot(relevantScans(1:12),relevantPeaks(1:12),'ro');
        plot(relevantScans(12:end),relevantPeaks(12:end),'kx');
        plot(relevantScans,regline,lineTypes{p},...
           'DisplayName',['Slope = ',num2str(coeffs(1)*1000),' mV/s']);
        %Use curve fitting toolbox for an error estimate
        try
        ccf = fit(relevantScans,relevantPeaks,'poly1');
        ccf_confint = confint(ccf);
        slp_uncert = (ccf_confint(2,1) - ccf_confint(1,1))*1000/2;
        fprintf('Slope for Peak #%i = %0.2f +/- %0.3f mV/s \n',p,coeffs(1)*1000,slp_uncert);
        end
       end
    end
    set(gca,'FontSize',14)
    xlabel('log_{10}[Scan Rate] (log_{10}[V/s])');
    ylabel('Inflection Potential vs Ag/AgCl (V)');
    legend('Location','West','FontSize',10);
    title(infoStruct.species);
end

end

function CVplot(b,NoExps,dataMat,traceNo)
    figure(b+NoExps); hold on;
    OCPs = find(dataMat{b,1}==dataMat{b,1}(end));
    if OCPs(1) ~= 1
        OCPs = [1;OCPs];
    end
    if dataMat{b,1}(end) > 2.5 %The Gamry Correction
        OCPs = [1;0;length(dataMat{b,1})];
    end
    if traceNo == 0
        ScanNo = floor(length(OCPs)/2);
        ScanVect = 1:ScanNo;
    else
        ScanVect = traceNo;
    end
    colors = {'r','g','b','k','c','m','y'};
    for i = ScanVect
        lowdex = 1 + 2*(i-1);
        highdex = lowdex + 2;
        dexvect = OCPs(lowdex):OCPs(highdex);
        traceV = dataMat{b,1}(dexvect); 
        traceI = dataMat{b,2}(dexvect);
        plot(traceV,traceI,colors{i},'LineWidth',0.5,'DisplayName',['Trace #',num2str(i)]);
    end
        
    xlim([min(dataMat{b,1}),max(dataMat{b,1})]);
    xlabel('Potential vs. Ag/AgCl in ACN (V)');
    ylabel('Current(A)');
    title(sprintf('Scan rate: %0.4f V/s',dataMat{b,3}));
    legend('Location','NorthWest');
    set(gca,'FontSize',14)  
end