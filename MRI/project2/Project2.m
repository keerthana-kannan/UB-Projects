%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        CROSS-RANGE IMAGING         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

cj=sqrt(-1);
pi2=2*pi;

c=3e8;                % propagation speed
fc=200e6;           % frequency
lambda=c/fc;         % Wavelength
k=pi2/lambda;       % Wavenumber
Xc=2e3;                 % Range distance to center of target area

L=150;
Y0=200;
Yc=500;

theta_c=atan(Yc/Xc);   % squint angle to the center of target area
Rc=sqrt(Xc^2+Yc^2);   % squint range to center of target area
kus=2*k*sin(theta_c);  % Doppler frequency shift in ku domain due to squint, i.e. center

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   Program performs slow-time compression to save PRF   %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Xcc=Xc/(cos(theta_c)^2);           % redefine Xc by Xcc for squint processing

du=(Xcc*lambda)/(4*(Y0+L));    % determined by support band of s(w,u)
duc=(Xcc*lambda)/(4*Y0);         % determined by support band of sc(w,u), the compressed signal
                           

L_min=max(Y0,L);                     % Zero-padded aperture

% u domain parameters and arrays for compressed signal sc(w,u)

mc=2*ceil(L_min/duc);                % number of samples on aperture s(w,u).
uc=duc*(-mc/2:mc/2-1);            % synthetic aperture array
dkuc=pi2/(mc*duc);                    % sample spacing in ku domain, 2pi=duc*dkuc*mc. 
kuc=dkuc*(-mc/2:mc/2-1);         % kuc array: compressed signal is base-band signal

dku=dkuc;                                   % sample spacing in ku domain, dku=dkuc=pi/L_min, L_min = Y0.

% u domain parameters and arrays for Synthetic aperture signal s(w,u)

m=2*ceil(pi/(du*dku));            % number of samples on aperture 
du=pi2/(m*dku); 
u=du*(-m/2:m/2-1);               % synthetic aperture array
ku=dku*(-m/2:m/2-1);           % ku array 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%          SIMULATION      %%%%%%%%%%          
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntarget=4;             % Number of targets

% Targets' coordinates and reflectivity

yn(1)=0;             fn(1)=1; % Y: yn in [-Y0, Y0].
yn(2)=.7*Y0;         fn(2)=0.8;
yn(3)=.6*Y0;         fn(3)=1;
yn(4)=-0.8*Y0;       fn(4)=0.6;

s=zeros(1,mc);         % Measured SAR Signal 
for i=1:ntarget;
     dis=sqrt(Xc^2+(Yc+yn(i)-uc).^2);                % origin is at (Xc,Yc), 
     s=s+fn(i)*exp(-cj*2*k*dis).*(abs(uc) <= L); % outside |L| is for 0-padding.
     s1=s; %passband signal
end;
% s has origin at (Xc,Yc), and the center freq. can be computed by inst.
% freq. in u domain, and that is kus.
s=s.*exp(-cj*kus*uc);     % Slow-time baseband conversion for squint 

fs=fty(s);

%Compression
sc=s1.*exp(cj*2*k*sqrt(Xc^2+(Yc-uc).^2));
fsc=fty(sc);
y=(kuc)*Rc/(2*k*cos(theta_c))+Yc;



figure(1) 
plot(uc,real(s),'b-'); grid on; % baseband
xlabel('Synthetic Aperture u (m)')
ylabel('Re[s_b(w,u_{ic})]'); % Y: sampled at duc, but not compressed yet.
title('P2.1 Baseband Aliased Synthetic Aperture Signal');
axis([uc(1) uc(mc) 1.1*min(real(s)) 1.1*max(real(s))]);
axis('square');
set(gca,'fontsize',16);

figure(2) 
plot(kuc,abs(fs),'b-'); grid on; 
xlabel('ku (m)')
ylabel('|S_b(w,u_{ic})|'); 
title('P2.2 Baseband Aliased Synthetic Aperture Signal Spectrum');
axis([kuc(1) kuc(mc) 1.1*min(abs(fs)) 1.1*max(abs(fs))]);
axis('square');
set(gca,'fontsize',16);

figure(3) 
plot(uc,real(sc),'b-'); grid on; 
xlabel('Synthetic Aperture u (m)')
ylabel('Re[s_c(w,u_{ic})]'); 
title('P2.3 Compressed Synthetic Aperture Signal');
axis([uc(1) uc(mc) 1.1*min(real(sc)) 1.1*max(real(sc))]);
axis('square');
set(gca,'fontsize',16);

figure(4) 
plot(kuc,abs(fsc),'b-'); grid on; 
xlabel('ku (m)')
ylabel('|S_c(w,k_u{ic})|');
title('P2.4a Compressed Synthetic Aperture Signal Spectrum');
axis([kuc(1) kuc(mc) -0.5*min(abs(fsc)) 1.5*max(abs(fsc))]);
axis('square');
set(gca,'fontsize',16);

figure(5) 
plot(y,abs(fsc),'b-'); grid on; 
xlabel('y')
ylabel('|S_c(w,k_u{ic})|'); 
title('P2.4b Compressed Synthetic Aperture Signal Spectrum');
axis([min(y) max(y) -0.5*min(abs(fsc)) 1.5*max(abs(fsc))]);
axis('square');
set(gca,'fontsize',16);



