function CV_Analysis6()
%CV_ANALYSIS5 - Uses semiderivative methods and more!
close all; clc;

%Assumes (so far) dataMat has already been allocated...
%Bring dataMat into the current directory
dataMat = evalin('base','dataMat');

%Toggles
onset = 1.0;
offset = 3; 
FTN = 25;
%11-20 are experiments and 1 - 10 are their blanks
%expDexes = [11,12,13,14,15,16,17,18,19,20]; %100 mV/s (11) to 10 V/s (17)
expDexes = 11;%[11,12,13,14,15];
bkgDexes = expDexes-10;
%expDexes = 2:13;
%bkgDexes = ones(1,length(expDexes));
%expDexes = 14:26;
%bkgDexes = expDexes - 13;

H = getHCPs();

response = input('Derivative analysis (1) or Semiderivative analysis (2)? ');
if response == 1
    fprintf('Derivative analysis selected.\n');
    derivAnalysis(onset,offset,FTN,expDexes,bkgDexes,H,dataMat);
else
    fprintf('Semiderivative analysis selected.');
    semiDerivAnalysis(onset,offset,FTN,expDexes,bkgDexes,H,dataMat);
end

end

function derivAnalysis(onset,offset,FTN,expDexes,bkgDexes,H,dataMat)
    closeMode = 1;
    subtraction = 0;
    exiting = 0;
    cvPlotting = 0;
    response = input('Background subtraction? y/n ','s');
    if strcmp(response,'y')
       subtraction = 1; 
    end
    response = input('Plot CVs? y/n ','s');
    if strcmp(response,'y')
       cvPlotting = 1; 
    end
    for qr = 1:length(expDexes)
          %Get the experiment from the dataMat
        experiment = dataMat(expDexes(qr),:);
        %Extract voltages, currents, and the scan rate
        voltages = experiment{:,1};
        currents = experiment{:,2};
        nu = experiment{3};
        %Get an aligned final trace
        [exprV,exprI] = getFTA(voltages,currents);
        %Create a background-subtracted signal
        if subtraction == 1
            %Extract background voltages and currents
            background = dataMat(bkgDexes(qr),:);
            bvoltages = background{:,1};
            bcurrents = background{:,2};
            %Get an aligned background trace
            [~,bkgI] = getFTA(bvoltages,bcurrents);
            %Background subtract
            try
                exprI = exprI - bkgI;
            catch
                fprintf('background could not be subtracted. \n');
            end
        end
        if cvPlotting == 1
           figure(qr+length(expDexes)); hold on;
           plot(exprV,exprI,'-b','LineWidth',1.5);
           xlabel('Voltage vs Ag/AgCl (V)');
           ylabel('Current (A)');
           title([num2str(nu,3),'V/s']);
           set(gca,'FontSize',16);
        end
    [oxidV,oxidI] = getOxidPortion(exprV,exprI,onset,offset);
    [Ideriv,Vderiv] = propDeriv(oxidI,oxidV,FTN);
    [I2deriv,V2deriv] = propDeriv(Ideriv,Vderiv,FTN);
    Ideriv = Ideriv(3:end-2); 
    figure(qr); hold on;
    xlabel('Potential vs. Ag/AgCl in ACN (V)');
    ylabel('Normalized (to max) Current Deriv. (-)');
    title(sprintf('Scan rate: %0.4f V/s',nu));
    set(gca,'FontSize',14)
    plot(V2deriv,Ideriv,'-k','LineWidth',1,'DisplayName','Deriv');
    ylim([-0.1,2.5]);
    yyaxis right
    ylabel('Normalized (to max) Current 2nd Deriv. (-)');
    ylim([-2.5,1]);
    set(gca,'ycolor','b');
    plot(V2deriv,I2deriv,'-b','LineWidth',1,'DisplayName','2nd Deriv');
    plot([V2deriv(1),V2deriv(end)],[0,0],'--r','LineWidth',1,'DisplayName','2nd Deriv Zero');
    
    fprintf('Examine mode. <> Indicates commands. Current scan: %0.4f V/s. \n',nu);
    fprintf('Type <help> for a list of commands. \n');
    cursor = imdistline(gca,[2,2],[-10,10]);
    setLabelVisible(cursor,0);
    doneExamine = 0;
    while doneExamine == 0
        response = input('Input command: ','s');
        response = lower(response);
        if strcmp(response,'help')
           fprintf('<next> - go to next scan. \n'); 
           fprintf('<peak> - place peak. \n');
           fprintf('<up#>/<down#> - vertical zoom in/out on #th derivative (1 or 2). \n');
           fprintf('<xin>/<xout> - zooms in x axis at cursor location/resets zoom. \n')
           fprintf('<closing> - enters/exits closing mode (only current figures visible). \n');
        elseif strcmp(response,'exit')
            doneExamine = 1;
            exiting = 1;
        elseif strcmp(response,'next')
            fprintf('Moving to next scan. \n');
            if closeMode == 1
                figure(qr);
                close
                figure(qr+length(expDexes));
                close
            end
            doneExamine = 1;
        elseif strcmp(response,'peak')
                peakVolt = getPosition(cursor);
                peakVolt = peakVolt(1,1);
                fprintf('Cursor Voltage: %0.10f \n',peakVolt);
        elseif strcmp(response,'up1') || strcmp(response,'up2') || strcmp(response,'down1') || strcmp(response,'down2')
                if strcmp(response,'up1') || strcmp(response,'down1')
                    yyaxis left
                    limits = ylim;
                    if strcmp(response,'up1')
                       limits = limits*0.25;
                    else
                       limits = limits*4;
                    end
                else
                    yyaxis right
                    limits = ylim;
                    if strcmp(response,'up2')
                       limits = limits*0.25;
                    else
                       limits = limits*4;
                    end
                end 
                ylim(limits)
        elseif strcmp(response,'xin')||strcmp(response,'xout')
            if strcmp(response,'xin')
                    peakVolt = getPosition(cursor);
                    peakVolt = peakVolt(1,1);
                    xlim([peakVolt-0.25,peakVolt+0.25]);
            else
                    xlim([onset,offset]);
            end
        elseif strcmp(response,'closing')
            if closeMode == 0
                fprintf('Close mode activated. \n');
                closeMode = 1;
            else
                fprintf('Close mode disabled. \n');
                closeMode = 0;
            end
        else
            fprintf('Command not understood. \n');
        end
        if doneExamine == 0
            figure(qr);
        end
    end
        if exiting == 1
            break
        end
    end
end


function semiDerivAnalysis(onset,offset,FTN,expDexes,bkgDexes,H,dataMat)
%Part #1 - prepare relevant matrices    
    for qr = 1:length(expDexes)
    %Get the experiment from the dataMat
    experiment = dataMat(expDexes(qr),:);
    %Extract voltages, currents, and the scan rate
    voltages = experiment{:,1};
    currents = experiment{:,2};
    nu = experiment{3};
    %Get an aligned final trace
    [exprV,exprI] = getFTA(voltages,currents);
    %Extract background voltages and currents
    background = dataMat(bkgDexes(qr),:);
    bvoltages = background{:,1};
    bcurrents = background{:,2};
    %Get an aligned background trace
    [~,bkgI] = getFTA(bvoltages,bcurrents);
    %Create a background-subtracted signal
    exprCI = exprI-bkgI;
    %Perform the 0.5 semiderivative on the whole CV
    %max(exprCI(exprV == 2.261))
    semiDCI = fourierSemiDeriv(exprV,exprCI,0.5);
    %Retrieve the oxidative portion of the CV
    [oxidVD,semiDCIC] = getOxidPortion(exprV,semiDCI,onset,offset);
    %Smooth the signal
    semiDCIC = smooth(oxidVD,semiDCIC,FTN);
    %Store this stuff in the matrix and show it off
    figure(2); hold on;
    plot(oxidVD,semiDCIC,'-k');
    if qr == 1
        oxidVDMat = zeros(length(oxidVD),length(expDexes));
        semiDCICMat = zeros(length(oxidVD),length(expDexes));
        nuVect = zeros(length(expDexes),1);
    end
    oxidVDMat(:,qr) = oxidVD;
    semiDCICMat(:,qr) = semiDCIC;
    nuVect(qr) = nu;
    end
    
%Part 2 - try a fit to semiderivative data
    %Lambda (for one peak) = (preF,Ep,alph,slp)
    %Complete lambda is [Lambda1,Lambda2,Lambda3, etc]
    figure(2); hold on;
    %plot(oxidVD,semiDCIC,'-k');
    %Assign # of peaks
    NN = 4;
    %Assign initial guesses for the peaks...
    Lambda0 = makeGuess(NN);
    %Create the constraints...
    [lb,ub,A,B,Aeq,Beq] = getConstraints1(NN,onset,offset,H);
    redSigEvalMult = @(L) manySignalEval(NN,nuVect,L,oxidVDMat,semiDCICMat,H);
    tol = 1e-12; opts = optimoptions('fmincon','Display','off','TolX',tol,...
        'TolFun',tol,'MaxFunctionEvaluations',1e6);
    [Lambda,fval] = fmincon(redSigEvalMult,Lambda0,A,B,Aeq,Beq,lb,ub,@nonlcon,opts);
    reportLambda(Lambda,nu,H);
    nnn = length(oxidVD).*length(expDexes);
    BIC = nnn*log(fval./nnn) + H.Nparam*NN*log(nnn);
    fprintf('Final Cost: %0.3e, BIC = %0.3e \n',...
        fval,BIC);
    figure(2); hold on;
    for qq = 1:length(nuVect)
        signal = makeSignal(NN,nuVect(qq),Lambda,oxidVDMat(:,qq),H);
        plot(oxidVDMat(:,qq),signal,'-r');
        for qop = 1:NN
           signal = makeSignal(1,nuVect(qq),...
               Lambda((1 + H.Nparam*(qop-1)):(H.Nparam + H.Nparam*(qop-1))),...
               oxidVDMat(:,qq),H);
           plot(oxidVDMat(:,qq),signal,'--b');
        end
    end
    
    %Plot niceties
    figure(2); hold on;
    set(gca,'FontSize',14)
    xlabel('Voltage vs. Ag/AgCl (V)');
    ylabel('Semideriv Signal (A/s^{1/2})');
    %ylim([0,20*10^-4]);

end

function Ltab = reportLambda(Lambda,nu,H)
    NN = length(Lambda)/H.Nparam;
    kV = 0:(NN-1);
    preF = Lambda(kV*H.Nparam + 1)';
    Ep_at100 = Lambda(kV*H.Nparam + 2)';
    alph = Lambda(kV*H.Nparam + 3)';
    slp = Lambda(kV*H.Nparam + 4)';
    Ltab = table(preF,Ep_at100,alph,slp);
    disp(Ltab);
    fprintf('Scan rate = %0.3f V/s \n',nu);
end

function Lambda0 = makeGuess(NN)
    Lambda01 = [1e-4,2.10,0.5,30];
    Lambda0 = Lambda01;
    if NN > 1
    Lambda02 = [1e-4,2.34,0.5,30];
    Lambda0 = [Lambda0,Lambda02];
    end
    if NN > 2
    Lambda03 = [1e-4,2.65,0.5,30];
    Lambda0 = [Lambda0,Lambda03];
    end
    if NN > 3
    Lambda04 = [1e-4,2.81,0.5,30];
    Lambda0 = [Lambda0,Lambda04];
    end
    if NN > 4
    Lambda05 = [1e-4,2.50,0.5,30];
    Lambda0 = [Lambda0,Lambda05];
    end
end

function [c,ceq] = nonlcon(Lambda)
    c = zeros(size(Lambda));
    ceq = [];
end

function [lb,ub,A,B,Aeq,Beq] = getConstraints1(NN,onset,offset,H)
    A = []; B = []; Aeq = []; Beq = [];
    lb = zeros(NN*H.Nparam,1);
    ub = lb;
    k = 0:(NN-1);
    %k*H.Nparam... +1 = preF, +2 = Ep, +3 = alph, +4 = slp 
    %preFactor low = 0 (infinitely slow diffusion)
    %preFactor high = (2*2 electrons)*(2 stoichiometry)*(2 follow up)*...
    %(sqrt(1e-8) m^2/s diffusion) = approx 0.002; 
    ub(k*H.Nparam + 1) = 0.002; %see above
    lb(k*H.Nparam + 1) = 0; 
    ub(k*H.Nparam + 2) = offset; %Initially the peaks should be present at 100 mV/s
    lb(k*H.Nparam + 2) = onset; %1 to 3 V is a good bound
    ub(k*H.Nparam + 3) = 1; %Transfer coefficient between 0 and 1
    lb(k*H.Nparam + 3) = 0;
    lb(k*H.Nparam + 4) = 15; %15 mV/dec (because, eh, why not)
    ub(k*H.Nparam + 4) = 200; %200 mV/dec (because, also, eh, why not)
end

function f = manySignalEval(NN,nuVect,Lambda,oxidVMat,semiDCICMat,H)
    f = 0;
    for qq = 1:length(nuVect)
       oxidV = oxidVMat(:,qq);
       semiDCIC = semiDCICMat(:,qq);
       f = f + signalEval(NN,nuVect(qq),Lambda,oxidV,semiDCIC,H);
    end
end

function f = signalEval(NN,nu,Lambda,oxidV,semiDCIC,H)
    signal = makeSignal(NN,nu,Lambda,oxidV,H);
    f = sum((signal-semiDCIC).*(signal-semiDCIC))/sqrt(nu);
end

function signal = makeSignal(NN,nu,Lambda,oxidV,H)
    signal = zeros(size(oxidV));
    for k = 0:(NN-1)
        preF = Lambda(k*H.Nparam + 1);
        Ep = Lambda(k*H.Nparam + 2);
        alph = Lambda(k*H.Nparam + 3);
        if nu ~= 0.1
            deltanu = log10(nu) - log10(0.1);
            slp = Lambda(k*H.Nparam + 4);
            Ep = Ep + (slp/1000)*deltanu;
        end
        scfcn = makePeakFcn(nu,preF,Ep,alph,0,H);
        signal = signal + scfcn(oxidV);
    end
end

function pfcn = makePeakFcn(nu,preF,Ep,alph,RR,H)
    %nu = scan rate
    %n = # of electrons transferred = 1 (assumed)
    %si = stoichiometry (1 or 2, probably)
    %eef = secondary electron factor (1 + 1*# of secondary electrons),
    %probably less than 2
    %D = diffusion coefficient
    %preF = n*n*si*eef*sqrt(D) (reversible) 
    %Ep = peak potential
    %alph = charge transfer coeff
    %RR = 1 if reversible, 0 if irreversible
    %H = hardcoded parameters
    if RR == 1
       %Make reversible function
       PF = 0.25*H.SA*H.F*H.F*H.Cb*nu*preF/(H.R*H.T);
       eF = 0.5*1*H.F/(H.R*H.T);
       pfcn = @(V) PF./(cosh(eF.*(V-Ep)).*cosh(eF.*(V-Ep)));
    else
       %Make irreversible function
        ep = 0.297*H.SA*H.F*H.F*H.Cb*sqrt(nu)*preF/(H.R*H.T);
        wp = 2.94*H.R*H.T./(alph*H.F);
        PR1 = ep/1.047;
        PR4 = exp(-1*(log(wp) - 0.353));
        PR3 = Ep + 0.171/PR4;
        PR2 = 1.544*PR4; %Result from M Grden - confirmed by Dalrymple-Alford
        pfcn = @(V) 4.*PR1./((exp(PR2.*(V - PR3)) + exp(-1*PR4.*(V-PR3))).^2);
    end
end

function yP = fourierSemiDeriv(x,y,p)
    DT = x(2) - x(1);
    N = length(x) - 1;
    m = (0:(N))'; %m(1) = 1e-10;
    Y = fft(y);
    Q = -(1i*2/(DT)).*sin(pi.*m/N).*(cos(pi.*m/N) + 1i.*sin(pi.*m/N));
    %Easy fix! (OK??)
    Q(1) = 1e-15;
    Yp = (Q.^p).*Y;
    yP = real(ifft(Yp,'symmetric'));
end

function [oxidV,oxidI] = getOxidPortion(voltages,currents,onset,offset)
    %Get all currents and voltages at voltages greater than the onset
    oxidI = currents(voltages>onset);
    oxidV = voltages(voltages>onset);
    %If an offset isn't specified
    if isempty(offset)
        %Go to the maximum voltage and make that the offset
        [~,enddex] = max(oxidV);
    else
       %Set the maximum voltage as the offset
       [~,enddex] = min(abs(oxidV-offset)); 
    end
    %Grab only the 'first part' of the trace (forward sweep)
    oxidI = oxidI(1:enddex);
    oxidV = oxidV(1:enddex);
end

function [ftV,ftI,OCPs] = getFTA(voltages,currents)
    %Get Final Trace Aligned
    %Finds the open circuit voltages and reorganizes so that the currents
    %and voltages correspond between traces with different OCPs
    %Find open circuit points....
    OCPs = find(voltages==voltages(end));
    %Grab the final trace (last three OCPs)...
    pre_ftV = voltages(OCPs(end-2):OCPs(end));
    pre_ftI = currents(OCPs(end-2):OCPs(end));
    %Find where the voltage equals zero...
    zeroPts = find(pre_ftV == 0);
    %CV = first zero -> next zero -> last point -> back to first zero
    reindexer = [(zeroPts(1):(zeroPts(2)-1))';((zeroPts(2)):length(pre_ftV))';(1:(zeroPts(1)-1))'];
    %Use this neat reindexing to get the aligned CVs
    ftV = pre_ftV(reindexer);
    ftI = pre_ftI(reindexer);
end

function [dI,corrV] = propDeriv(I,V,FTN)
    %Produces a smoothed finite difference derivative, removes end points.
    %Derivative is also normalized to its max value
    dI = (I(1:end-4)*(1/12) - I(2:end-3)*(2/3) ...
         + I(4:end-1)*(2/3) - I(5:end)*(1/12));
    corrV = V(3:end-2); 
    dI = smooth(corrV,dI,FTN);
    dI = dI/max(abs(dI));
end

function H = getHCPs()
    %Hardcoded parameters function
    H.T = 298; %K - "Lab temp"
    H.F = 96485; %C/mol - Faraday's const.
    H.SA = (0.5*3/1000)*(0.5*3/1000)*pi; %m^2 - RDE
    H.Cb = 1; %mol/m^3 - 1 mM
    H.R = 8.314; %J/mol-K - Gas constant
    H.Nparam = 4; %# of parameters in Lambda
end
