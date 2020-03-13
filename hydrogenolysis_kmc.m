function hydrogenolysis_kmc()
%boilerplate, begin timing
close all; clear; clc;
tic

%Set initial system parameters
c_init = 18; %Initial carbon chain length
N_init = 5000; %Initial # of molecules
M = 200; %Number of simulations to run
Xspec = 0.73; %Specified reaction conversion (end of simulation)
OverTON = 0; %Specified additional turnovers past 100% conversion. 
%To use turnover count only, set Xspec to zero. 
T = 300; %System temperature (K)
H2Ratio = 1000*c_init; %Initial # of H2 molecules/# of N_init
boreno = 1;

%Set system physics (turn one parameter to 1, else to 0)
%Which molecules react? Uniform distribution for nonzero molecs by default.
moleculeWeighting = 1;
carbonWeighting = 1;
rateWeighting = 0;
hydrogenWeighting = 0;
%Where does the C-C bond cleave? Several options...
%Option #1: set proportional to 1, then set relative prop. of special
%events. "else" event is random cutting. (ex: 0.25 half, 0.25 terminal ->
%0.5 random cutting). 
proportional = 0;
halfcutting = 0;
%terminalcutting = 0.3;
terminalcutting = 0.8;
%Option #2: uses custom distribution. set custom to 1, then set sp. dist.
custom = 1;
%Any custom probability distribution can be used once decided upon.
%bias towards center
centralcutting = 0;
%bias towards center with secondary bias towards terminal cutting
centralcuttingwithterm = 1;
%relative probability increase for terminal cuts
termfactor = 20;

%Set isomerization option - tracks linear and 'isomerized' species
isomerizing = 0;

%Set relative rates of adsorption/rxn to the catalyst:
%rates are in vector [ethane, propane, butane... C-N_init]
selRates = ((2:c_init).^2).*T; 
%Currently - all rates linearly associated with T - rate ~ size^2. 
%Isomers are currently just given designation 'isomer' for simplicity.
%Set isomerization rate (likelihood that any given rxn is an isomerization)
isoRate = 0.05;
deIsoRate = 0.25;
%Set overall reactivity of isomer species
%iso ads rates are in vector [isopropane, isobutane,...isoC-N_init]
isoSelRates = selRates(2:end);
%set if feed is isomer
isoFeed = 0;
%Set hydrogen effect factor (scales rates)
H2EffectFactor = 0.0001;

%Other options
%Spec plot: specify carbon # of interest to see distribution of results
%over M experiments. turn spec plot to 1 to see results + data readout.
specNo = 10;
specOutput = 0;
%Bucket by... (# of carbons)
bucketSpace = c_init/10;
%Julie Plot?
juliePlot = 1;
JCOL = [0.1,0.2,0.9];
%JCOL = [0.1,1,0.2];
rng default

%The "states" vector: each unit represents # of molecules, index
%represents carbon # of those molecules (linear alkanes only)
%Initialize vector for tracking states across M simulations
%If isomerizing, state vector is extended by c_init-2 elements
if isomerizing == 1
    stateavg = zeros(1,(2*c_init-2));
    %[methane,ethane,...,C-N_init,isopropane,isobutane,...isoC-N_init]
else
    stateavg = zeros(1,c_init);
    %[methane,ethane,...,C-N_init]
end

%Various pre-experiment calculations...
if isomerizing == 1
    redVect = [2:c_init,3:c_init];
    selRates = [selRates,isoSelRates];
    redIndexing = redVect;
    redIndexing(c_init:end) = redIndexing(c_init:end)+c_init-2;
else
    redVect = 2:c_init;
    redIndexing = redVect;
end
spec = zeros(1,M);
tvals = zeros(1,M);
H2Count = H2Ratio*N_init;
H2Feed = H2Count;

%For each experiment...
for j = 1:M
    %Initialize the experiment-specific states vector...
    states = 0*stateavg;
    if isoFeed == 1
        states(end) = N_init;
    else
        states(c_init) = N_init;
    end
    OverTONct = 0;
    tinc = 0;
    %While the feed has not conv. to degree Xspec and completed OverTON...
    while states(c_init) > (1-Xspec)*N_init || OverTONct < OverTON
        %If specified conversion is met, increment OverTurnovers
        if states(c_init) <= (1-Xspec)*N_init
           OverTONct = OverTONct + 1;
        end
        if sum(states(2:end)) == 0
            break
        end
        %relative weights of rxn likeliness for molecular/carbon weighting
        unWtProb1 = double(states(2:end)>0);
        if moleculeWeighting == 1
            unWtProb1 = unWtProb1.*states(2:end);
        end
        if carbonWeighting == 1
           unWtProb1 = unWtProb1.*redVect; 
        end
        if rateWeighting == 1
           unWtProb1 = unWtProb1.*selRates; 
        end
        if hydrogenWeighting == 1
           unWtProb1 = unWtProb1*H2Count*H2EffectFactor;
        end
        %Normalize probability distribution, select molecule from dist
        WtProb1 = unWtProb1./sum(unWtProb1);
        ind = SelectFrom(WtProb1,redIndexing);
    %Compute avg rxn time...    
        %Need to think about this + metathesis problem.
        invRateSum = 1/sum(states(2:end).*selRates);
        tinc = tinc + invRateSum*log10(1/rand());
    %Test if molecule isomerizes
        if isomerizing ==1 && ind > c_init && rand() < deIsoRate
            %use de-isomerization rate for test
             states(ind) = states(ind) - 1;
             states(ind-c_init+2) = states(ind-c_init+2) + 1;
        elseif isomerizing ==1 && ind > 2 && ind<= c_init && rand()<isoRate
            %use isomerizing rate for test
            states(ind) = states(ind) - 1;
            states(ind + c_init-2) = states(ind + c_init-2) + 1;
        else
            %Cutting event!
            if proportional == 1
                ucup = rand();
                if ucup < halfcutting
                    %cut at center
                    point = ceil(ind/2);
                elseif ucup > halfcutting && ucup < (halfcutting + terminalcutting)
                    %cut at end
                    point = 1;
                else
                    %random cut
                    point = ceil(rand()*(ind-1)); 
                end
            elseif custom == 1
                if centralcutting == 1
                    %custom distribution - prefer near-center cuts
                    cutValues = 1:(ind-1);
                    unWtProb2 = 1./(abs(cutValues-1).^2 + abs(ind-cutValues).^2);
                    WtProb2 = unWtProb2./sum(unWtProb2);
                    point = SelectFrom(WtProb2,cutValues);
                end
                if centralcuttingwithterm == 1
                   cutValues = 1:(ind-1);
                   unWtProb2 = 1./(abs(cutValues-1).^5 + abs((ind-1)-cutValues).^5);
                   unWtProb2(1) = unWtProb2(1)*termfactor;
                   unWtProb2(end) = unWtProb2(1);
                   WtProb2 = unWtProb2./sum(unWtProb2);
                   if ind == 2
                      WtProb2 = 1;
                      cutValues = 2;
                   end
                   %Warning: will crash matlab if run not in debugger
                   if boreno == 1
                     figure; hold on;
                     plot(WtProb2,'rx','MarkerSize',10,'LineWidth',2);
                     plot(WtProb2,'r-','LineWidth',2);
                     xlabel('C-C Bond #','FontSize',20); 
                     ylabel('Cut Probability','FontSize',20);
                     ax = gca;
                     ax.FontSize = 16;
                      boreno = 0;
                   end
                   point = SelectFrom(WtProb2,cutValues);
                end
            end
            %After selecting point, remove the cleaved parent chain...
            states(ind) = states(ind) - 1;
            %... and add in the two new shorter daughter chains.
            states(point) = states(point) + 1;
            if ind ~= 2
            states(ind-point) = states(ind-point) + 1;
            else
            states(1) = states(1) + 2; 
            end
            %Update the global hydrogen concentration accordingly
            H2Count = H2Count - 2;
        end
    end
    %End of simulation
    if j == 1
        %If this is the first experiment, the average is simply the states
        stateavg = states;
        spec(j) = states(specNo);
        tvals(j) = tinc;
    else
        %If this is a subsequent experiment, the average is the prev. avg.
        %averaged with the new states
        stateavg = (stateavg + states)*0.5;
        tvals(j) = tinc;
        spec(j) = states(specNo);
    end
end

%Generate carbon distribution plot
figure; hold on;
xlabel('Carbon #');
ylabel('Frequency (C/C)');
linstates = stateavg(1:c_init)/sum(stateavg);
if c_init < 50
    bar(1:(c_init),stateavg(1:c_init).*(1:c_init)/sum(stateavg.*(1:c_init)));
else
    inc = 1;
    brng1 = 1;
    brng2 = bucketSpace;
    lingroups = zeros(1,(floor(c_init/bucketSpace)-1));
    while brng2 <= c_init
    lingroups(inc) = sum(linstates(brng1:brng2));
    linnumlabels(1,inc) = brng1;
    linnumlabels(2,inc) = brng2;
    linlabels{inc} = [num2str(brng1),'-',num2str(brng2)];
    inc = inc + 1;
    brng1 = brng1 + bucketSpace;
    brng2 = brng2 + bucketSpace;
    end
    if brng1 < c_init
       lingroups(inc) = sum(linstates((brng1):end));
       linnumlabels(1,inc) = brng1;
       linnumlabels(2,inc) = brng2;
       linlabels{inc} = [num2str(brng1),'-',num2str(c_init)];
    else
       inc = inc - 1; 
    end
    bar(lingroups);
    set(gca,'xticklabel',linlabels);
    ax = gca;
    ax.XTick = 1:length(linlabels);
    xtickangle(45)
end
if isomerizing == 1
    isostates = stateavg((c_init+1):end)/sum(stateavg);
    if c_init < 50
        bar(3:c_init,stateavg((c_init+1):end)/sum(stateavg));
        legend('n-alkanes','i-alkanes','Location','North');
    else
        isogroups = zeros(1,length(lingroups));
        for L = 1:inc
            if linnumlabels(2,L) < length(isogroups)
            isogroups(L) = sum(isostates(linnumlabels(1,L):linnumlabels(2,L))); 
            else
            isogroups(L) = sum(isostates(linnumlabels(1,L):end));    
            end
        end
        bar(isogroups);
    end
end

%Generate error-analysis plot
if specOutput == 1
    figure; hold on;
    histogram(spec);
    xlabel('Molecule Ct.','FontSize',20);
    ylabel('Frequency','FontSize',20);
    ax = gca;
    ax.FontSize = 16;
    fprintf('==================begin spec diagnostics=================\n');
    fprintf('spec C avg count: %0.4f, std: %0.4f, rel. error is %0.4f of feed \n',...
    mean(spec),std(spec),(3*std(spec))/N_init);
    fprintf('H2 remaining/initial: %0.4f \n',H2Count/H2Feed);
    fprintf('carbon balance average/feed: %0.4f \n',...
    sum(stateavg.*[1,redVect])/(c_init*N_init));
    fprintf('avg. rxn time: %0.4e, std: %0.4e \n',mean(tvals),std(tvals));
    fprintf('==================end spec diagnostics===================\n');
end
if juliePlot == 1
   figure; hold on;
   xlabel('Carbon #','FontSize',20);
   ylabel('Frequency (C/C)','FontSize',20);
   lincarbonstates = stateavg(1:c_init).*(1:c_init)./sum(stateavg(1:c_init).*(1:c_init));
   liquidNormal = lincarbonstates(7:17)./sum(lincarbonstates(7:17));
   %bar(7:17,lincarbonstates(7:17),'FaceColor',JCOL);
   bar(7:17,liquidNormal,'FaceColor',JCOL);
   xlim([6.5,17.5])
   totalC = sum(lincarbonstates);
   liquidC = sum(lincarbonstates(7:end))./totalC;
   %gasMoles = 1 - liquidMoles;
   str1 = sprintf('X_{C18}: %0.2f, C_7-C_{18} Fraction: %0.2f',Xspec,liquidC);
   %str2 = sprintf('C7-C18 Fraction = %0.2f',liquidMoles);
   %str3 = sprintf('C1-C6 Fraction = %0.2f',gasMoles);
   %txt = {str1,str2,str3};
   title(str1,'FontSize',16)
   ax = gca;
   ax.FontSize = 16;
end

%End timing
toc
end

function [index,u] = SelectFrom(weightedProb,ProbVals)
    %sub-function that selects from the weighted probability distribution
    %probability distribution is given from C2->...->CN
    %likely that CN is more common than C2, so probabilities and values
    %are flipped for easier coding
    wPP = flip(weightedProb);
    PV = flip(ProbVals);
    %Generate random number, set "done" toggle
    u = rand(); done = 0;
    %Test simplest case (for a well organized wPP)
    if u < wPP(1) && wPP(1) ~= 0
        index = PV(1);
        done = 1;
    end
    %Walk up through all cases to find where the probability sits
    if done == 0
        summer = wPP(1);
        k = 1;
        while done == 0
                if u > summer && u < (summer + wPP(k+1)) && wPP(k+1) ~=0
                    index = PV(k+1);
                    done = 1;
                end
            try    
            summer = summer + wPP(k+1);
            catch
            fprintf('problem');
            end
            k = k + 1;
        end
    end
end