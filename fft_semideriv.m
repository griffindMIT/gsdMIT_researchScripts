function yP = fft_semideriv(x,y,p)
%Function written by Griffin Drake, April 2020 (version 1)
%Algorithm is based on Yu, J.S., et al, J. Electroanal Chem, v. 403, 1996.
%"Differentiation, semidifferentiation and semi-integration of a digital
%signals based on Fourier transformations."

%Function takes evenly-spaced, discrete x,y data and returns the p'th
%derivative. For p =/= integer, this is a weighted convolution between
%integer derivatives. See the reference for more information. 

%Inputs:    x = vector of x-values for periodic signal data
%           y = vector of y-values for periodic signal data
%           p = # of derivative. p > 0, derivative. p < 0, integral. 

    %Compute spacing, # of points from x data. 
    DT = x(2) - x(1);
    N = length(x) - 1;
    %Compute index # for operator
    m = (0:(N))'; 
    %transform y data
    Y = fft(y);
    %Create operator for p'th derivative
    Q = -(1i*2/(DT)).*sin(pi.*m/N).*(cos(pi.*m/N) + 1i.*sin(pi.*m/N));
    %Correct first entry (prevents divide by zero error for p < 0)
    Q(1) = 1e-15;
    %Perform p'th differentiation
    Yp = (Q.^p).*Y;
    %Return signal to real space
    yP = real(ifft(Yp,'symmetric'));
end

