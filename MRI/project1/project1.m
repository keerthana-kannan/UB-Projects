
clear all
close all
cj=sqrt(-1);
pi2=2*pi;
c=3e8;                        % Propagation speed                
B0 = 50e6; % Hz               % Baseband bandwidth
w0=pi2*B0; % rad/s
fc=1e9;                       % Carrier frequency: Hz
wc=pi2*fc; % rad/s
Xc=2.e3;                     % Range distance to center of target area
Tp = 10e-6;   % [0, Tp]

alpha=w0/Tp;                 % Chirp rate % Y
beta=wc-alpha*Tp;         % % start point of frequency beta = wcm = wc -w0. [beta, beta+2*alpha*Tp]
                                       % carrier freq. of the chirp is the
                                       % mid-freq. wc = beta + alpha*Tp,
                                       % and the spectral support band is
                                       % [beta, beta+2alpha*Tp]
                                       % p(t) = a(t)*exp(j*beta*t+j*alpha*t^2).
                                       % wi(t) = beta*t+2*alpha*t, t>=0,
                                       % alpha>0.

dx=c/(4*B0);                 % Range resolution % x = c*t/2, dx = c*dt/2
dt=pi/(2*alpha*Tp);      % Time domain sampling (guard band plus minus
                                     % 50 per % Y: ws>=4w0) or use dt=1/(2*B0) for a general
                                     % radar signal % dt = 1/(2*B0)=pi/w0. dt means
                                     % Delta_t                        
Tx=0.67e-6;                % Range swath echo time period % Y: Xc-X0<=x<=Xc+X0
X0 =(c*Tx)/4;                     % target area in range is within [Xc-X0,Xc+X0]
dtc=pi/(2*alpha*Tx);    % Time domain sampling for compressed signal
                                     % (guard band plus minus 50 per) % Y: dtc means
                                     % Delta_{tc}
Ts=(2*(Xc-X0))/c;          % Start time of sampling % Y
Tf=(2*(Xc+X0))/c+Tp;   % End time of sampling % Y

% If Tx < Tp, choose compressed signal parameters for measurement
flag=0;                  % flag=0 indicates that Tx > Tp
if Tx < Tp,
   flag=1;                 % flag=1 indicates that Tx < TP
   dt_temp=dt;             % Store dt % Y
   dt=dtc;                 % Choose dtc (dtc > dt) for data acquisition
end;

% Measurement parameters
n=2*ceil((.5*(Tf-Ts))/dt);        % Number of time samples % Y: n is even
t=Ts+(0:n-1)*dt;                    % Time array for data acquisition, ti
dw=pi2/(n*dt);                       % Frequency domain sampling % Y: fs = 1/dt is the bandwidth
w=wc+dw*(-n/2:n/2-1);        % Frequency array (centered at carrier) 
x=Xc+.5*c*dt*(-n/2:n/2-1);   % range bins (array); reference signal is
                                               % for target at x=Xc. % Y: t=2x/c, so x =
                                               % tc/2.
kx=(2*w)/c;                            % Spatial (range) frequency array % Y
x= (((w-wc)*c)/(4*alpha))+Xc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ntarget=5;                        % number of targets
%%%%%%%%%%%%% Targets' parameters  %%%%%%%%%%%%%%%%%%

% xn: range;    % Y: relevant shift against Xc           fn: reflectivity

xn(1)=0;                   fn(1)=1;
xn(2)=.7*X0;               fn(2)=.8;
xn(3)=.8*X0;               fn(3)=1;
xn(4)=-.5*X0;              fn(4)=.8;
xn(5)=.3*X0;               fn(5)=.7;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s=zeros(1,n);              % Initialize echoed signal array

for i=1:ntarget;
     td=t-2*(Xc+xn(i))/c; % Y: td are the actual delayed time points, t includes the delay of the closest reflector
     pha=beta*td+alpha*(td.^2);         
     % Chirp (LFM) phase 
     % Y: beta = wc-w0 is the modified chirp carrier freq.
     s=s+fn(i)*exp(cj*pha).*(td >= 0 & td <= Tp); % dtc, s is aliased.
     % Y: td is the new time points, only when 0<=td<=Tp, p(td)~=0.
end

% If flag=1, i.e., Tx < Tp, perform upsampling
if     flag == 1
       td0=t-2*(Xc+0)/c;
       pha0=beta*td0+alpha*(td0.^2); % Reference chirp phase 
       % Baseband compressed signal                               
       scb = conj(s).*exp(cj*pha0).*exp(cj*2*beta*Xc/c-cj*4*alpha*Xc^2/(c^2)); % scb: alias-free
       fscb = fty(scb);
       
figure(8)
subplot(1,1,1)
plot(x,(dt/Tp)*abs(fscb),'b-'); grid on;
%xlim = ([1950 2050])
axis([Xc-X0 Xc+X0 0 max((dt/Tp)*abs(fscb))])
xlabel('x')
ylabel('|s_c_b(w)|')
title('Range reconstruction via time domain compression');
%axis('square')
%axis([Ts Tf min(abs(fsb)) 1.1*max(abs(fsb))])
set(gca,'fontsize',12);
       
       scb=[scb,scb(n:-1:1)];  % Append mirror image in time to reduce wrap 
                             % around errors in interpolation (upsampling)
       fscb=fty(scb);          % F.T. of compressed signal w.r.t. time

       dt=dt_temp;                     % Time sampling for echoed signal
       n_up=2*ceil((.5*(Tf-Ts))/dt);   % Number of time samples for upsampling
       nz=n_up-n;                      % number of zeros for upsampling is 2*nz % Y
       fscb=(n_up/n)*[zeros(1,nz),fscb,zeros(1,nz)]; 

       scb=ifty(fscb);
       scb=scb(1:n_up);            % Remove mirror image in time % Y: now scb is the upsampled signal.

       % Upsampled parameters
       n=n_up;
       t=Ts+(0:n-1)*dt;             % Time array for data acquisition
       dw=pi2/(n*dt);                % Frequency domain sampling
       w=wc+dw*(-n/2:n/2-1);        % Frequency array (centered at carrier)
       x=Xc+.5*c*dt*(-n/2:n/2-1);   % range bins (array); reference signal is
                                              % for target at x=Xc.
       kx=(2*w)/c;                    % Spatial (range) frequency array
       s=conj(scb).*exp(cj*beta*t+cj*alpha*t.^2-cj*4*alpha*Xc*t/c);
end

% Baseband conversion
sb=s.*exp(-cj*wc*t); 
fsb = fty(sb);

%Matched Filtering
td0=t-2*(Xc+0)/c;
pha0=beta*td0+alpha*(td0.^2);
s0= exp(cj*pha0); %reference signal
s0b= s0.*exp(-cj*wc*t);    %baseband conversion
fs0b= fty(s0b);            % frequency domain
fsmb= conj(fs0b).*fsb;          %Baseband matched filtering
smb= ifty(fsmb);          % f(x)


figure(1)
subplot(1,1,1)
plot(t,real(sb),'b-'); grid on;
xlabel('t (s)')
ylabel('Re[s_b(t)]')
title('Baseband Echoed Signal');
axis('square')
axis([Ts Tf min(real(sb)) 1.1*max(real(sb))])
set(gca,'fontsize',18);

figure(2)
subplot(1,1,1)
plot(t,(dt/Tp)*real(s0b),'b-'); grid on;
xlabel('t (s)')
ylabel('Re[s_0_b(t)]')
title('Baseband Reference Echoed Signal');
axis('square')
axis([Ts Tf 2*min((dt/Tp)*real(s0b)) 2*max((dt/Tp)*real(s0b))])
%axis([Ts Tf min((dt/Tp)*real(s0b)) 1.1*max((dt/Tp)*real(s0b))])
set(gca,'fontsize',14);

figure(3)
subplot(1,1,1)
plot(w-wc,abs(fsb),'b-'); grid on;
xlabel('w')
ylabel('|s_b(w)|')
title('Baseband-echoed signal Spectrum');
%axis('square')
%axis([-7 7 min(abs(fsb)) 1.1*max(abs(fsb))])
set(gca,'fontsize',18);

figure(4)
subplot(1,1,1)
plot(w-wc,abs(fs0b),'b-'); grid on;
xlabel('w')
ylabel('|s_0_b(w)|')
title('Baseband-echoed Reference signal Spectrum');
axis('square')
%axis([Ts Tf min(abs(fsb)) 1.1*max(abs(fsb))])
set(gca,'fontsize',18);

figure(5)
subplot(2,1,1)
plot(w-wc,abs(fsmb),'b-'); grid on;
xlabel('w')
ylabel('|s_m_b(w)|')
title('Baseband matched filtered signal spectrum');
axis('square')
%axis([Ts Tf min(abs(fsb)) 1.1*max(abs(fsb))])
set(gca,'fontsize',14);

kx=(2*(w-wc))/c; 
subplot(2,1,2)
plot(kx,abs(fsmb),'b-'); grid on;
xlabel('kx')
ylabel('|F(kx)|')
title('Baseband matched filtered signal spectrum');
axis('square')
%axis([Ts Tf min(abs(fsb)) 1.1*max(abs(fsb))])
set(gca,'fontsize',14);

figure(6)
subplot(2,1,1)
plot(t,real(smb),'b-'); grid on;
%xlim([1.4*10e-6 2.2*10e-6])
%ylim([min() 60])
xlabel('t (s)')
ylabel('Re[s_m_b(t)]')
title('Range reconstruction via baseband matched filtering');
%axis('square')
axis([Ts Tf min(real(smb)) 1.1*max(real(smb))])
set(gca,'fontsize',12);

subplot(2,1,2)
plot(x,abs(smb),'b-'); grid on;
xlim([1900 2100])
ylim([0 2200])
xlabel('x')
ylabel('|f(x)|')
title('Range reconstruction via baseband matched filtering');
%axis('square')
%axis(1500 2500 min(abs(smb)) 1.1*max(abs(smb))])
set(gca,'fontsize',12);


figure(7)
subplot(1,1,1)
plot(t,real(scb),'b-'); grid on;
xlabel('t (s)')
ylabel('Re[s_c_b(t)]')
title('Baseband Compressed Signal');
axis('square')
axis([Ts Tf min(real(scb)) 1.1*max(real(scb))])
set(gca,'fontsize',18);

