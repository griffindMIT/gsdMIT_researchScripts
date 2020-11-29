function hydrogenolysis_kmc()
%boilerplate, begin timing
close all; clear; clc; tic;

%System parameters =======================================================
c_ctr = 18; %Initial carbon chain length/distribution center.
mState = 0; %0 if input state is all c_ctr, 1 if input is a distribution
    c_dev = 50; %Deviation from c_ctr if multistate is used (0 else)
    c_skew = 1; %Skew applied to c_ctr if multistate is used (0 else)
monMw = 20; %Monomer molecular weight (amu)
N = 5000; %Initial # of molecules
M = 50; %Number of simulations to run
Xs = 0.7; %Specified reaction conversion of c_ctr!
OverTON = 0; %Specified additional turnovers past Xspec. Xs = 0 for TON only.
Tsys = 500; %System temperature (K)
H2Rat = 10000; %Initial # of H2 molecules/# of N_init
%=========================================================================

%Simulation toggles (simple) ======================================================
molecWtOrd = 1; %Concentration rate order (all compounds)
carbonWtOrd = 1; %If 0, no effect, 1, apply size weighting to molecules
hydgOrd = 0; %Overall rate order for hydrogen gas effect - set 0 if mech-specific is desired
%=========================================================================

%Simulation toggles (distrib.) ==========================================
method_frac = 1; %Turn on 'fractional method.' 
frac_rand = 0; %Proportion for random cutting 
frac_term = 0; %Proportion for terminal cutting
frac_cust = 1; %Proportion for custom distribution. all fracs sum to 1!
if (frac_rand + frac_term + frac_cust) ~= 1 
    fprintf('Bad fractional toggles. \n');
end
cust_dist = @centralPower; %Function handle for relative weight as f(C-C) bond
%========================================================================

%Plotting toggles =======================================================
displayIn = 1; %Shows C-C distribution for c_init species
normtyp = 1; %(0) is by mol, (1) is by carbon #
specNo = 10; %Specify C# whose variance is of interest. If not interested, 0
norm_type = 1; %Use (0) for mol normalizing, (1) for carbon normalizing.
%"Julie Plot" focuses on liquid regions. 0 if not desired, [r,g,b] if so.
JCOL = [0.1,0.2,0.9];
%Figures: (1) Input (2) Custom cut dist. (3) Raw output (4/5) Spec/H2 (6) Jplot
%========================================================================

%Pre-calculations/allocations =====================================================
rng default %Reset RNG
[feedState,statVals,c_max] = genInput(c_ctr,c_dev,c_skew,N,mState,monMw,displayIn); %Generate input state
stateavg = 0.0*feedState; %set an 'average' vector which will change over M simulation
accStat = 2:c_max; %Vector of 'accessible' molecules to cut (excludes methane) 
spec = zeros(1,M); %Storage for values of 'spec' carbon
H2Feed = H2Rat*N; %Sets qty. of H2 at system start
H2Fin = 0; %Sets storage variable for final H2 quantity
H2spec = zeros(1,M); %Storage for value of 'spec' H2
%========================================================================

%Main computation loop ==================================================
for j = 1:M %For each experiment...
    states = feedState; %Initialize working 'states' vector
    OverTONct = 0; %Reset turnover count
    H2Count = H2Feed; %Initialize H2 in chamber
    while states(c_ctr) > (1-Xs)*N || OverTONct < OverTON %c_ctr not converted & overTON not reached       
        %Check that reactivity is valid (molecules present / no (-) concs.)
        if sum(states(2:end)) == 0 %Overdraft protection
            fprintf('Insufficient funds - exiting. \n');
            break
        elseif sum(states<0) > 0
            fprintf('Overdrawn - exiting. \n');
            break
        end
        
        %Computation 1 - molecule selection. 
        unWtP1 = double(states(accStat)>0); %Initialize equal-wt for all present molecules
        unWtP1 = unWtP1.*(states(accStat).^molecWtOrd); %Concentration order
        unWtP1 = unWtP1.*(accStat.^carbonWtOrd); %Carbon size dependence
        unWtP1 = unWtP1.*(H2Count.^hydgOrd); %Hydrogen effect
        %Normalize probability distribution, select molecule from dist
        WtP1 = unWtP1./sum(unWtP1);
        ind = SelectFrom(WtP1,accStat); %ind is index & molecule size

        %Computation 2 - molecule cutting. Selects 'point' (bond to be cut)
        if method_frac == 1 %'fractional' method
            u_mech = rand(); %Random # for mechanism selection
            if u_mech <= frac_rand %Random scission
                point = ceil(rand()*(ind-1)); 
            elseif u_mech > frac_rand && u_mech <= (frac_term+frac_rand) %Terminal scission
                point = 1; %Currently this assumes symmetry.
            elseif u_mech > (frac_term+frac_rand) %Select from custom distro
                cutVals = 1:(ind-1);
                WtP2 = cust_dist(ind,cutVals);
                %Display / saving code here
                if displayIn == 1 && ind == c_max %Rep. display of largest input
                     figure(2); hold on;
                     %Need code here about custom shite
                     plot(WtP2,'r-','LineWidth',2);
                     xlabel('C-C Bond #','FontSize',20); 
                     ylabel('Cut Probability','FontSize',20);
                     ax = gca;
                     ax.FontSize = 16;
                     displayIn = 0;
                end
                point = SelectFrom(WtP2,cutVals);   
            end
        else %nonfractional methods
            fprintf('Not currently supported. \n'); 
            break
        end
        
        %Computation 3: Post-cleave updates
        states(ind) = states(ind) - 1; %Remove parent
        states(point) = states(point) + 1; %Add daughter #1
        if ind ~=2 %Add the other daughter (with caution for methane)
            states(ind-point) = states(ind-point)+1;
        else
            states(1) = states(1) + 1;
        end
        H2Count = H2Count - 2; %Update H2 accordingly
        %Increments OverTON with bool - only adds if X >= xspec
        OverTONct = OverTONct + (states(c_ctr) <= (1-Xs)*N);
    end
    
    if j == 1
        %Exp #1 - The average is the result.
        stateavg = states;
        H2Fin = H2Count; 
    else
        %Exp # 2 to M - The average is the average.
        stateavg = (stateavg + states)*0.5;
        H2Fin = (H2Fin + H2Count)*0.5;
    end
    spec(j) = states(specNo); %Store spec information
    H2spec(j) = H2Count/H2Feed;
end
%====================================================================

%Post-computation presentation =======================================
fullPlot(stateavg,statVals,normtyp); %Generic carbon dist plot
fprintf('Average H2Fin/H2Feed: %0.4f \n',H2Fin/H2Feed); %Report H2 consumed
specOutput(specNo,spec,H2spec,stateavg,feedState,statVals)%Spec output
JuliePlot(JCOL,stateavg,statVals,Xs)%Julie Plot
toc %End timing
%===================================================================
end

function WtP2 = centralPower(ind,cutVals)
    %A distribution function
    unWtP2 = 1./(abs(cutVals).^6 + abs(ind-cutVals).^6); %needs fixed
    WtP2 = unWtP2./sum(unWtP2);
end

function fullPlot(stateavg,statVals,normtyp)
    figure(3); hold on;
    set(gca,'FontSize',16);
    xlabel('Carbon #');
    ylabel('Frequency (C/C)');
    if normtyp == 0
        normed = stateavg/sum(stateavg); %mol_norm
    elseif normtyp == 1
        normed = (stateavg.*statVals)./sum(stateavg.*statVals);
    end
    bar(statVals,normed);
end

function specOutput(tog,spec,H2spec,stateavg,feedState,statVals)
    if tog ~=0
        figure(4); hold on;
        histogram(spec);
        xlabel('Molecule Ct.');
        ylabel('Frequency');
        set(gca,'FontSize',16);
        fprintf('Spec C avg count: %0.4f, std: %0.4f, Rel. error is %0.4f of feed \n',...
        mean(spec),std(spec),(3*std(spec))/sum(feedState)); 
        fprintf('Carbon balance average/feed: %0.4f \n',...
        sum(stateavg.*statVals)/sum(feedState.*statVals));
        fprintf('Remaining H2 mean: %0.4f, std: %0.4f, Rel. error is %0.4f of feed \n',...
        mean(H2spec),std(H2spec),3*std(H2spec));
        figure(5); hold on; 
        histogram(H2spec);
        xlabel('H2Fin/H2Feed');
        ylabel('Frequency');
        set(gca,'FontSize',16);
    end
end

function JuliePlot(JCOL,stateavg,statVals,Xs)
    if JCOL ~= 0
        figure(6); hold on;
        xlabel('Carbon #');
        ylabel('Frequency (C/C)');
        set(gca,'FontSize',16);
        carbon_norm = (stateavg.*statVals)./sum(stateavg.*statVals);
        liquid_norm = carbon_norm(7:end)./sum(carbon_norm(7:end));
        bar(7:length(carbon_norm),liquid_norm,'FaceColor',JCOL);
        xlim([6.5,length(carbon_norm)-0.5])
        totalC = sum(carbon_norm);
        liquidC = sum(carbon_norm(7:17));
        str1 = sprintf('X_{C18}: %0.2f, C_7-C_{18} Fraction: %0.2f',Xs,liquidC/totalC);
        title(str1)
    end
end


function [feedState,stateVals,c_max] = genInput(ctr,dev,skew,N,mState,monWt,disppt)
    if mState == 1 %a distribution input
        cvals = (ctr-6*dev):1:(ctr+6*dev); %possible carbon values considered (six sigmas)
        cvals = cvals(cvals>0); %correct to only positive carbon values
        pvals = normpdf(cvals,ctr,dev).*normcdf(skew.*(cvals-ctr)./dev); %generate skewed gaussian
        pvaln = pvals./sum(pvals); %normalize probs
        feedState = [0*(1:(cvals(1)-1)),round(pvaln*N,0)]; %Create state vector
        exists_values = find(feedState>0); %identify nonzero components
        c_max = exists_values(end); %id largest c value
        feedState = feedState(1:exists_values(end)); %remove zero feed states (trimming)
    else
        feedState = [0*(0:1:(ctr-2)),N]; %Feed state is mono
        c_max = ctr; %Input size is max
    end
    stateVals = 1:length(feedState); %useful vector to have around
    if disppt == 1
        Mn = sum(monWt.*stateVals.*feedState)./sum(feedState);
        Mw = sum(monWt*monWt*stateVals.*stateVals.*feedState)./sum(monWt*stateVals.*feedState);
        figure(1); plot(stateVals,feedState,'.b','LineWidth',2,'MarkerSize',20);
        fprintf('Input dist: Mn = %0.3f, Mw = %0.3f \n',Mn,Mw);
        set(gca,'FontSize',16);
        xlabel('Carbon #');
        ylabel('Molecule Count');
    end
end

function [index,u] = SelectFrom(wPP,PV)
    %Sub-function that selects from weighted prob distribution (wPP) and
    %returns corresponding probability value (PV)
    u = rand(); %Generate random number (selected)
    cwPP = cumsum(wPP); %Cumulative distribution
    index = PV(find(u<cwPP,1)); %First index u<cwPP == upper bound of reg.
    %u is also returned for debugging purposes
end
