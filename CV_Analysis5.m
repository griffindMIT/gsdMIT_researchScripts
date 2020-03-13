function CV_Analysis5()
%CV_ANALYSIS4 - Uses semiderivative methods
close all; clc;

%Assumes (so far) dataMat has already been allocated...
%Bring dataMat into the current directory
dataMat = evalin('base','dataMat');

%Toggles
onset = 1.5;
offset = 3; 
FTN = 25;
numba = 11;
H = getHCPs();

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
    %Lambda1 = (n,D,Ep,alph,si,RR) - highest possible parameter set
    figure(2); hold on;
    plot(oxidVD,semiDCIC,'-k');
    NN = 4; rng default; 
    %Hold structure = value, lambda index, 0/1//lb/ub
    hold_RR = [0,6,0;0,6,1]; 
    hold_si = [1,5,3;1,5,3];    
    holds = [hold_RR];
    %TestLambda1 = [1,1e-9,2.1,0.5,1,1];
    %TestLambda2 = [1,0.5*1e-9,2.6,0.2,1,0];
    %plot(oxidVD,makeSignal(NN,nu,TestLambda1,oxidVD,H));
    %plot(oxidVD,makeSignal(NN,nu,TestLambda2,oxidVD,H));
    [IntCon,lb,ub,A,B,Aeq,Beq] = getConstraints1(NN,onset,max(oxidVD),holds);
    redSignalEval = @(L) signalEval(NN,nu,L,oxidVD,semiDCIC,H);
    opts = optimoptions('ga','Display','iter','ConstraintTolerance',1e-13,'FunctionTolerance',1e-13,'MaxGenerations',1000*NN*3);
    tic
    Lambda = ga(redSignalEval,NN*6,A,B,Aeq,Beq,lb,ub,@nonlcon,IntCon,opts);
    toc
    reportLambda(Lambda);
    signal = makeSignal(NN,nu,Lambda,oxidVD,H);
    figure(2); plot(oxidVD,signal,'-b');
    for joj = 0:(NN-1)
       redLambda = Lambda(((joj*6)+1):((joj*6)+6));
       signal = makeSignal(1,nu,redLambda,oxidVD,H);
       plot(oxidVD,signal,'-r');
    end
    
end

function Ltab = reportLambda(Lambda)
    NN = length(Lambda)/6;
    kV = 0:(NN-1);
    n = Lambda(kV*6 + 1)';
    D = Lambda(kV*6 + 2)';
    Ep = Lambda(kV*6 + 3)';
    alph = Lambda(kV*6 + 4)';
    si = Lambda(kV*6 + 5)';
    RR = Lambda(kV*6 + 6)';
    Ltab = table(n,D,Ep,alph,si,RR);
    disp(Ltab);
end

function [c,ceq] = nonlcon(Lambda)
    c = zeros(size(Lambda));
    ceq = [];
end

function [IntCon,lb,ub,A,B,Aeq,Beq] = getConstraints1(NN,onset,offset,holds)
    A = []; B = []; Aeq = []; Beq = [];
    lb = zeros(NN*3,1);
    ub = lb;
    k = 0:(NN-1);
    %k*6... +1 = n, +2 = D, +3 = Ep, +4 = alph, +5 = stoich, +5 = RR. 
    ub(k*6 + 1) = 2; %1 or 2 electron transfers
    lb(k*6 + 1) = 1; 
    ub(k*6 + 2) = 1e-7; %m^2/s, & this is wicked fast diffusion
    lb(k*6 + 3) = onset; %only voltages in traces
    ub(k*6 + 3) = offset;
    lb(k*6 + 4) = 0; %transfer coeff between 0 and 1
    ub(k*6 + 4) = 1;
    lb(k*6 + 5) = 1; %considering only 1 or 2 w.r.t. bulk concentration
    ub(k*6 + 5) = 2;
    lb(k*6 + 6) = 0; %Zero or 1 for reversibility toggle 
    ub(k*6 + 6) = 1;
    % (+,1,+5,+6) are ints.
    IntCon = [(k*6 + 1),(k*6 + 5),(k*6 + 6)];
    if ~isempty(holds)
        for kk = 1:length(holds(:,1))
            %holds is an ? x 3 matrix, ? is the # of imposed holds
            %col 1 = hold value, col 2 is Lambda index, col 3 0/1 lb/ub 
            if holds(kk,3) == 0
                lb(k*6 + holds(kk,2)) = holds(kk,1); 
            else
                ub(k*6 + holds(kk,2)) = holds(kk,1); 
            end
        end
    end
end

function Lambda_full = LambdaExpander(NN,Lambda,Lred,Lfix)
    %Lred = indices of which Lambda components are fixed
    %Lfix = if VECTOR => col = value corresponding to fix in Lred order
           %if MATRIX => col = as above, row = value for each peak
end

function f = signalEval(NN,nu,Lambda,oxidV,semiDCIC,H)
    %Lambda_full = LambdaExpander(Lambda);
    signal = makeSignal(NN,nu,Lambda,oxidV,H);
    f = sum((signal-semiDCIC).*(signal-semiDCIC));
end

function signal = makeSignal(NN,nu,Lambda,oxidV,H)
    signal = zeros(size(oxidV));
    for k = 0:(NN-1)
        n = Lambda(k*6 + 1);
        D = Lambda(k*6 + 2);
        Ep = Lambda(k*6 + 3);
        alph = Lambda(k*6 + 4);
        si = Lambda(k*6 + 5);
        RR = Lambda(k*6 + 6);
        scfcn = makePeakFcn(nu,n,D,Ep,alph,si,RR,H);
        signal = signal + scfcn(oxidV);
    end
end

function pfcn = makePeakFcn(nu,n,D,Ep,alph,si,RR,H)
    %nu = scan rate
    %n = # of electrons transferred
    %D = diffusion coefficient
    %Ep = peak potential
    %alph = charge transfer coeff
    %si = stoich coeff
    %RR = 1 if reversible, 0 if irreversible
    %H = hardcoded parameters
    if RR == 1
       %Make reversible function
       PF = 0.25*H.SA*n*n*H.F*H.F*si*H.Cb*nu*sqrt(D)/(H.R*H.T);
       eF = 0.5*n*H.F/(H.R*H.T);
       pfcn = @(V) PF./(cosh(eF.*(V-Ep)).*cosh(eF.*(V-Ep)));
    else
       %Make irreversible function - a wee bit hardcoded...
       %Use "C" route
%        ep = 0.297*H.SA*n*n*H.F*H.F*si*H.Cb*nu*sqrt(D)/(H.R*H.T);
%        wp = 2.94*H.R*H.T./(n*alph*H.F);
%        PR1 = ep/1.047;
%        PR4 = exp(-1*(log(wp) - 0.353));
%        PR3 = Ep + 0.171/PR4;
%        PR2 = 0.89*PR4; %1.544*PR4;
%        pfcn = @(V) 4.*PR1./((exp(PR2.*(V - PR3)) + exp(-1*PR4.*(V-PR3))).^2);
       %Try another irreversible function....
       tc = 1:25;
       PF = -H.SA*alph*n*n*H.F*H.F*H.Cb*si*nu/(H.R*H.T);
       termPF = ((-1).^tc).*(tc).*sqrt((factorial(tc))).*exp(tc.*alph*H.F*n.*()./(H.R.*H.T))
       pfcn = @(V) PF.*();
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

function H = getHCPs()
    %Hardcoded parameters function
    H.T = 298; %K
    H.F = 96485; %C/mol
    H.SA = (3/1000)*(3/1000)*pi; %m^2 
    H.Cb = 1; %mol/m^3  
    H.R = 8.314; %J/mol-K
end