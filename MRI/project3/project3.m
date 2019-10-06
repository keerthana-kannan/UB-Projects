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

y=(kuc)*Rc/(2*k*cos(theta_c))+Yc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                       %% PROJECT3 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%upsampling by zero-padding

sc = [sc,sc(mc:-1:1)];
fsc = fty(sc);

mz= m-mc;
fsc = m/mc*[zeros(1,mz),fsc,zeros(1,mz)];
dku=dku*(-m:m-1);

sc=ifty(fsc);
sc=sc(:,1:m);

figure(1) 
plot(dku,abs(fsc),'b-'); grid on; 
xlabel('k_u (m)')
ylabel('|S_c(w,k_u)|'); 
title('P3.1 Compressed Synthetic Aperture Signal Spectrum');
%axis([-dku*m dku*m -0.5*min(abs(fsc)) 1.05*max(abs(fsc))]);
axis('square');
set(gca,'fontsize',16);


figure(2) 
plot(u,real(sc),'b-'); grid on; 
xlabel('u (m)')
ylabel('Re[s_c(w,u)]'); 
title('P3.2 Compressed Synthetic Aperture Signal');
axis([u(1) u(m) 1.1*min(real(sc)) 1.1*max(real(sc))]);
axis('square');
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%DECOMPRESSION IN u DOMAIN %%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

s=sc.*exp(-cj*2*k*sqrt(Xc^2+(Yc-u).^2));

sb=s.*exp(-cj*kus*u);

fsb=fty(sb);

figure(3) 
plot(u,real(sb),'b-'); grid on; 
xlabel('u (m)')
ylabel('Re[s_b(w,u)]'); 
title('P3.3 Alias free Baseband Synthetic Aperture Signal');
axis([u(1) u(m) 1.1*min(real(sb)) 1.1*max(real(sb))]);
axis('square');
set(gca,'fontsize',16);

figure(4) 
plot(ku,abs(fsb),'b-'); grid on; 
xlabel('k_u (m)')
ylabel('|S_b(w,k_u)|');
title('P3.4 Alias free Baseband Synthetic Aperture Signal Spectrum');
axis([ku(1) ku(m) 1.1*min(abs(fsb)) 1.5*max(abs(fsb))]);
axis('square');
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%  MATCHED FILTERING  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kx=4*k^2-(ku+kus).^2;
kx=sqrt(kx);

fs0b=exp((cj*kx*Xc)+cj*(ku+kus)*Yc);

fsm=fsb.*fs0b;

sm=ifty(fsm);

figure(5) 
plot(u+Yc,abs(sm),'b-'); grid on; 
xlabel('y (m)')
ylabel('|f(y)|');
title('P3.5 Cross-range Reconstruction');
axis([u(1)+Yc u(m)+Yc 1.1*min(abs(sm)) 1.5*max(abs(sm))]);
axis('square');
set(gca,'fontsize',16);












