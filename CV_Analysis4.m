function CV_Analysis4()
%CV_ANALYSIS4 - Uses semiderivative methods
close all; clc;

%Assumes (so far) dataMat has already been allocated...
%Bring dataMat into the current directory
dataMat = evalin('base','dataMat');

%Toggles
onset = 1.9;
offset = 2.9; 
FTN = 25;
numba = 11;

%Part #1 - load experiment & bkg (preprocessing)
    experiment = dataMat(numba,:);
    voltages = experiment{:,1};
    currents = experiment{:,2};
    nu = experiment{3};
    [exprV,exprI] = getFTA(voltages,currents);
    background = dataMat(numba-10,:);
    bvoltages = background{:,1};
    bcurrents = background{:,2};
    [bkgV,bkgI] = getFTA(bvoltages,bcurrents);
    figure(1); hold on;
    plot(exprV,exprI);
    plot(bkgV,bkgI);
    exprCI = exprI-bkgI;
    plot(exprV,exprCI,'-k');
    yyaxis right
    semiDCI = fourierSemiDeriv(exprV,exprCI,0.5);
    plot(exprV,semiDCI,'-r');
    [oxidVD,semiDCIC] = getOxidPortion(exprV,semiDCI,onset,offset);
    semiDCIC = smooth(oxidVD,semiDCIC,FTN);
    plot(oxidVD,semiDCIC,'-b');
    
%Part 2 - try a ga fit to semiderivative data
    figure(2); hold on;
    plot(oxidVD,semiDCIC,'-k');
    NN = 4;
    guessEhs = [2.1,2.3,2.6,2.2];
    vtol = 0.05;
    [IntCon,lb,ub,A,B,Aeq,Beq] = getConstraints(NN,onset,max(oxidVD),guessEhs,vtol);
    redCoshishEval = @(L) coshishEval(NN,nu,L,oxidVD,semiDCIC);
    opts = optimoptions('ga','Display','Off','ConstraintTolerance',1e-8,'FunctionTolerance',1e-8,'MaxGenerations',500*NN*3);
    rng default
    Lambda = ga(redCoshishEval,NN*3,A,B,Aeq,Beq,lb,ub,@nonlcon,IntCon,opts);
    Ltab = reportLambda(Lambda);
    disp(Ltab)
    signal = makeSignal(NN,nu,Lambda,oxidVD);
    figure(2);
    plot(oxidVD,signal,'-b');
    signal = makeSignal(1,nu,[1,1e-9,2.1],oxidVD);
    plot(oxidVD,signal,'-r');
    
%Part 3 - try a linearized fit to semiintegral data
%     figure(3); hold on;
%     semiICI = fourierSemiDeriv(exprV,exprCI,-0.5);
%     %plot(exprV,semiICI);
%     [oxidVI,semiICI] = getOxidPortion(exprV,semiICI,onset,offset);
%     semiICIC = smooth(oxidVI,semiICI,FTN);
%     semiICIC = semiICIC - semiICIC(1);
%     plot(oxidVI,semiICIC,'-k');
%     semiICIC_max = getLimm(1,1e-10)
%     plot([oxidVI(1),oxidVI(end)],[semiICIC_max,semiICIC_max],'--r')

%Part 4 - 
    
end

function Ltab = reportLambda(Lambda)
    NN = length(Lambda)/3;
    kV = 0:(NN-1);
    n = Lambda(kV*3 + 1)';
    D = Lambda(kV*3 + 2)';
    Eh = Lambda(kV*3 + 3)';
    Ltab = table(n,D,Eh);
end

function [c,ceq] = nonlcon(Lambda)
    c = zeros(size(Lambda));
    ceq = [];
end

function [IntCon,lb,ub,A,B,Aeq,Beq] = getConstraints(NN,onset,maxV,guessEhs,vtol)
    A = []; B = []; Aeq = []; Beq = [];
    lb = zeros(NN*3,1);
    ub = lb;
    IntCon = [];
    k = 0:(NN-1);
    ub(k*3 + 1) = 2;
    lb(k*3 + 1) = 1;
    ub(k*3 + 2) = 1e-7;
    IntCon = k*3 + 1;
    if isempty(guessEhs)
        lb(k*3 + 3) = onset;
        ub(k*3 + 3) = maxV;
    else
        lb(k*3 + 3) = guessEhs-vtol;
        ub(k*3 + 3) = guessEhs+vtol;
    end
end

function f = coshishEval(NN,nu,Lambda,oxidV,semiCIC)
    signal = makeSignal(NN,nu,Lambda,oxidV);
    f = sum((signal - semiCIC).*(signal - semiCIC));
end

function signal = makeSignal(NN,nu,Lambda,oxidV)
    signal = zeros(size(oxidV));
    for k = 0:(NN-1)
        n = Lambda(k*3 + 1);
        D = Lambda(k*3 + 2);
        Eh = Lambda(k*3 + 3);
        scfcn = makeSemicoshFcnHCRev(n,nu,D,Eh);
        signal = signal + scfcn(oxidV);
    end
end

function mInf = getLimm(n,D)
    A = (3/1000)*(3/1000)*pi; %m^2 - WARNING! HARDCODED!
    C = 1; %mol/m^3 - WARNING! HARDCODED! 
    F = 96485; %C
    mInf = n*A*F*C*sqrt(D);
end

function scfcn = makeSemiCoshFcnHCIrrev(n,nu,D,Eh)
    %D = m^2/s, nu = V/s, Eh = V, n = # of electrons    
    T = 298; %K - WARNING! HARDCODED!
    A = (3/1000)*(3/1000)*pi; %m^2 - WARNING! HARDCODED!
    C = 1; %mol/m^3 - WARNING! HARDCODED! 
    scfcn = makeSemiCoshFCNIrrev(n,nu,D,Eh,T,A,C);
end

function scfncn = makeSemiCoshFCNIrrev(n,nu,D,Eh,T,A,C)
    P1 = 0.297*A*n*n*F*F
end

function scfcn = makeSemicoshFcnHCRev(n,nu,D,Eh)
    %D = m^2/s, nu = V/s, Eh = V, n = # of electrons    
    T = 298; %K - WARNING! HARDCODED!
    A = (3/1000)*(3/1000)*pi; %m^2 - WARNING! HARDCODED!
    C = 1; %mol/m^3 - WARNING! HARDCODED! 
    scfcn = makeSemicoshFcnRev(n,nu,D,Eh,T,A,C);
end

function scfcn = makeSemicoshFcnRev(n,nu,D,Eh,T,A,C)
    F = 96485; %C
    R = 8.314; %J/mol-K
    PF = (n*n*F*F*A*nu*C*sqrt(D)/(4*R*T));
    ExpF = (n*F/(2*R*T));
    scfcn = @(E) PF./(cosh(ExpF.*(E-Eh)).*cosh(ExpF.*(E-Eh)));
end

function yP = fourierSemiDeriv(x,y,p)
    T = x(2) - x(1);
    N = length(x) - 1;
    m = (0:(N))'; %m(1) = 1e-10;
    Y = fft(y);
    Q = -(1i*2/(T)).*sin(pi.*m/N).*(cos(pi.*m/N) + 1i.*sin(pi.*m/N));
    %Easy fix!
    Q(1) = 1e-15;
    %Q(1) = 1e-0;
    Yp = (Q.^p).*Y;
    yP = real(ifft(Yp,'symmetric'));
end

function [oxidV,oxidI] = getOxidPortion(voltages,currents,onset,offset)
    oxidI = currents(voltages>onset);
    oxidV = voltages(voltages>onset);
    if isempty(offset)
        [~,enddex] = max(oxidV);
    else
       [~,enddex] = min(abs(oxidV-offset)); 
    end
    oxidI = oxidI(1:enddex);
    oxidV = oxidV(1:enddex);
end

function [ftV,ftI,OCPs] = getFTA(voltages,currents)
    %Get Final Trace Aligned
    OCPs = find(voltages==voltages(end));
    pre_ftV = voltages(OCPs(end-2):OCPs(end));
    pre_ftI = currents(OCPs(end-2):OCPs(end));
    zeroPts = find(pre_ftV == 0);
    reindexer = [(zeroPts(1):(zeroPts(2)-1))';((zeroPts(2)):length(pre_ftV))';(1:(zeroPts(1)-1))'];
    ftV = pre_ftV(reindexer);
    ftI = pre_ftI(reindexer);
end

function s = sig(x,a)
    s = exp(a.*x)./(exp(a.*x) + 1);
end

% %Some proof - of - concept stuff
% %Part #2 - load experiment (Pexin Processed)
%     experiment2 = dataMat(end,:);
%     voltages2 = experiment2{:,1};
%     currents2 = experiment2{:,2};   
%     [exprV2,exprI2] = getFTA(voltages2,currents2);
%     background2 = dataMat(end-1,:);
%     bvoltages2 = background2{:,1};
%     bcurrents2 = background2{:,2};
%     [bkgV2,bkgI2] = getFTA(bvoltages2,bcurrents2);
%     figure(2); hold on;
%     plot(exprV2,exprI2);
%     plot(bkgV2,bkgI2);
%     exprCI2 = exprI2-bkgI2;
%     plot(exprV2,exprCI2,'-k');
%     
%     figure(1); hold on;
%     exprCI1_recov = fourierSemiDeriv(exprV1,fourierSemiDeriv(exprV1,exprCI1,0.5),-0.5);
%     %Theoretically recoverable?
%     plot(exprV2,exprCI1_recov,'--r');
%     yyaxis right
%     plot(exprV1,fourierSemiDeriv(exprV1,exprCI1,0.5));
% 
%     figure(3); hold on;
%     x1 = -15:0.001:15;
%     x2 = x1-0.001; x2(1:2) = [];
%     x = [x1,flip(x2)];
%     y = sig(x,0.5);
%     plot(x,y,'-k');
%     figure(3);
%     y_sd = fourierSemiDeriv(x',y',0.5);
%     plot(x,y_sd,'-r');
%     y_d = fourierSemiDeriv(x',y',1);
%     plot(x,y_d,'-b');
%     y_fromd = fourierSemiDeriv(x',y_d,-1);
%     plot(x,y_fromd,'--b','LineWidth',2);
%     y_fromsd = fourierSemiDeriv(x',y_sd,-0.5);
%     plot(x,y_fromsd,'--r','LineWidth',3);
