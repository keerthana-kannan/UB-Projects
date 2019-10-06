clear all
close all
colormap(gray(256))
cj=sqrt(-1);
pi2=2*pi;

c=3e8;  
f0=50e6;
w0=pi2*f0;
fc=200e6;
wc=pi2*fc;
lambda_min=c/(fc+f0);
lambda_max=c/(fc-f0);
kc=(pi2*fc)/c;
kmin=(pi2*(fc-f0))/c; 
kmax=(pi2*(fc+f0))/c;

Xc=1000; 
X0=20;
Yc=300;
Y0=60;

%case1 L<Y0; requires zero padding of SAR signal in synthetic aperture
%domain
L=40;  %sythetic aperture is 2*L

theta_c=atan(Yc/Xc);
Rc=sqrt(Xc^2+Yc^2);
L_min=max(Y0,L);  %zeropadding in u domain

Xcc=Xc/(cos(theta_c)^2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%u domain paramteres and array for compressed SAR signal

duc=(Xcc*lambda_min)/(4*Y0); %sample spacing in aperture domain for compressed SAS signal

duc=duc/1.2;


mc=2*ceil(L_min/duc);
uc=duc*(-mc/2:mc/2-1);
dkuc=pi2/(mc*duc);
kuc=dkuc*(-mc/2:mc/2-1);  %kuc array

dku=dkuc;  %sample spacing in ku domain


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% u domain paramteres and arrays for SAR signal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if Yc-Y0-L<0
    theta_min=atan((Yc-Y0-L)/(Xc-X0));
else
    theta_min=atan((Yc-Y0-L)/(Xc+X0));
end
theta_max=atan((Yc+Y0+L)/(Xc-X0));
 %max aspect angle

kumin=min(2*kmin*sin(theta_min),2*kmax*sin(theta_min));  %ku=2*k*sin(theta)
kumax=2*kmax*sin(theta_max);
du=pi2/(kumax-kumin);

du=du/1.4;                    %20% gaurd band
m=2*ceil(pi/(du*dku));          %number of samples on aperture
du=pi2/(m*dku);            %readjust du
u=du*(-m/2:m/2-1);          %synthetic aperture array
ku=dku*(-m/2:m/2-1);               %ku array

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%Fast time domain parameters and array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Tp=2.5e-7;
alpha=w0/Tp;
wcm=wc-alpha*Tp;


if Yc-Y0-L<0
    Rmin=Xc-X0;
else
    Rmin=sqrt((Xc-X0)^2+(Yc-Y0-L)^2);
end
Ts=(2/c)*Rmin;
Rmax= sqrt((Xc+X0)^2+(Yc+Y0+L)^2);
Tf=(2/c)*Rmax+Tp;
T=Tf-Ts;
Ts=Ts-0.1*T;
Tf=Tf+0.1*T;
T=Tf-Ts;
Tmin=max(T,(4*X0)/(c*cos(theta_max)));

dt=1/(4*f0);
n= 2*ceil((0.5*Tmin)/dt);
t=Ts+(0:n-1)*dt;
dw=pi2/(n*dt);
w=wc+dw*(-n/2:n/2-1);
k= w/c;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Parameters of Target
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ntarget=9;

xn=zeros(1,ntarget); yn=xn;         fn=xn;

xn(1)=0;         yn(1)=0;            fn(1)=1;
xn(2)=0.7*X0;    yn(2)=-0.6*Y0;      fn(2)=1.4;
xn(3)=0;         yn(3)=-0.85*Y0;     fn(3)=0.8;
xn(4)=-0.5*X0;    yn(4)=0.75*Y0;     fn(4)=1;
xn(5)=-0.4*X0;    yn(5)=0.65*Y0;     fn(5)=1;
xn(6)=-1.2*X0;    yn(6)=0.75*Y0;     fn(6)=1;
xn(7)=0.5*X0;    yn(7)=1.25*Y0;      fn(7)=1;
xn(8)=1.1*X0;    yn(8)=-1.1*Y0;      fn(8)=1;
xn(9)=-1.2*X0;    yn(9)=-1.75*Y0;    fn(9)=1;

%%%%%%%%%%%%%%Simulation

s=zeros(n,mc);

for i=1:ntarget
    td=t(:)*ones(1,mc)-(2*ones(n,1)*sqrt((Xc+xn(i)).^2+(Yc+yn(i)-uc).^2)/c);
    s=s+fn(i)*exp(cj*wcm*td+cj*alpha*(td).^2).*(td>=0 & td <=Tp & ...
                                      ones(n,1)*abs(uc) <= L & t(:)*ones(1,mc)<Tf); 
   
end

s=s.*exp(-cj*wc*t(:)*ones(1,mc)); %Fast time baseband conversion
%for tddisplay
G=abs(s)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng); %x'=cg*(G-ng)

figure(1)   %P4.1
colormap(gray(256));
image(t,uc,256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Fast time t,sec');
ylabel('Synthetic aperture (slowtime) u, meters');
title('Measured spotlight SAR signal |s(t,u)|');
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%Fast time baseband matched filtering

Tc=2*Rc/c;
td0=t(:)-Tc;

s0=exp(cj*wcm*td0+cj*alpha*(td0.^2)).*(td0>=0 & td0<=Tp);

s0=s0.*exp(-cj*wc*t(:));

s=ftx(s).*(conj(ftx(s0))*ones(1,mc));  %column FT n*mc matrix.. This is sM(w,u)
%for display
G=abs(iftx(s))';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);
tm=Tc+dt*(-n/2:n/2-1);

figure(2)   %P4.2
colormap(gray(256));
image(tm,uc,256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Fast time t,sec');
ylabel('Synthetic aperture (slowtime) u, meters');
title('SAR signal after Fast time matched filtering |s_M(t,u)|');
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Slow time baseband conversion for squint
kus=2*kc*sin(theta_c)*ones(1,n);
s=s.*exp(-cj*kus(:)*uc);
fs=fty(s);

G=abs(fs)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(3)   %P4.2
colormap(gray(256));
image(k*c/pi2,kuc,256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Fast time frequency,Hertz');
ylabel('Synthetic aperture (slowtime) frequency k_u, rad/m');
title('Baseband Aliased Spotlight SAR signal spectrum |S(\omega,k_u)|'); %S(w,ku)
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Slow time compression

s=s.*exp(cj*kus(:)*uc);

cs= s.*exp(cj*2*(k(:)*ones(1,mc)).*(ones(n,1)*sqrt(Xc^2+(Yc-uc).^2))-cj*2*k(:)*Rc*ones(1,mc));
fcs=fty(cs);

G=abs(fcs)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(4)   %P4.4
colormap(gray(256));
image(k*c/pi2,kuc,256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Fast time frequency,Hertz');
ylabel('Synthetic aperture (slowtime) frequency k_u, rad/m');
title('Compressed Spotlight SAR signal spectrum |S_c(\omega,k_u)|'); %S(w,ku)
grid on
set(gca,'fontsize',16);

%%%Polar format processed reconstruction

fp=iftx(fty(cs));
 
%%Digital spotlighting (to suppress the echoed signal outside desired target area)

PH=asin(kuc/(2*kc));  %angular doppler domain
R=(c*tm)/2; 
%window func
W_d=((abs(R(:)*cos(PH+theta_c)-Xc)<X0).*(abs(R(:)*sin(PH+theta_c)-Yc)<Y0));

fd=fp.*W_d;  %digital spotlight filtering in (t,ku)
fcs=ftx(fd);  %Scd(w,ku)

%%Zero  padding in ku domain for slow time upsampling

mz=m-mc;
fcs=(m/mc)*[zeros(n,mz/2),fcs,zeros(n,mz/2)];

cs=ifty(fcs);

%slow time decompression  %sd(w,u)=scd(w,u)*s0(w,u)
s=cs.*exp(-cj*2*(k(:)*ones(1,m)).*(ones(n,1)*sqrt(Xc^2+(Yc-u).^2))+cj*2*k(:)*Rc*ones(1,m));

%%%%%%%%%%%%%%%%%%%%%%%

s_ds=s;

s=s.*exp(-cj*kus(:)*u);

fs=fty(s);

G=abs(fs)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(5)   %P4.5
colormap(gray(256));
image(k*c/pi2,ku,256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Fast time frequency,Hertz');
ylabel('Synthetic aperture (slowtime) frequency k_u, rad/m');
title('Spotlight SAR signal spectrum after DS & upsampling |S_d(\omega,k_u)|'); %DS=digital spotlighting
grid on
set(gca,'fontsize',16);

%===============================================

dky=dku;
ny=m;
ky=ku;
dy=pi2/(ny*dky);
y=dy*(-ny/2:ny/2-1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   Reconstruction

ky=ones(n,1)*ky+kus(:)*ones(1,ny); %ky =ku, n x m
kx=(4*k(:).^2)*ones(1,ny)-ky.^2;
kx=sqrt(kx.*(kx>0));  %kx array, n x ny

figure(6) %P4.6
colormap(gray(256));
plot(kx(1:20:n*ny),ky(1:20:n*ny),'.')
axis('square');axis('xy')
xlabel('Spatial frequency k_x, rad/m');
ylabel('Spatial frequency k_y, rad/m');
title('Spotlight SAR Spatial frequency data coverage'); 
axis image; axis xy
grid on
set(gca,'fontsize',16);


kxmin=min(min(kx));
kxmax=max(max(kx));
dkx=pi/X0;
nx=2*ceil((0.5*(kxmax-kxmin))/dkx);

%%%%%%%%%%%%%%%%%%%%%%%%%%% 2D macthed filtering and interpolation

fs0=(kx>0).*exp(cj*kx*Xc+cj*ky*Yc ...
    -cj*2*k(:)*ones(1,ny)*Rc);
fsm=fs.*fs0;

% ============================
% ========== Interpolation =========
% ============================
%function [F,KX,KY,kxc] = interpolation_code(fsm,kx,ky,dkx,nx,ny,kxmin,n)
% fsm is the 2D matched filtered F(kx,ky), unevenly spaced samples
is=8;       % number of neighbors (sidelobes) used for sinc interpolator, Ns
I=2*is+1;
kxs=is*dkx; % plus/minus size of interpolation neighborhood in KX domain, i.e. kxs = Ns*dkx
%
nx=nx+2*is+4;  % increase number of samples to avoid negative
                         %  array index during interpolation in kx domain
KX=kxmin+(-is-2:nx-is-3)*dkx;     % KX: uniformly-spaced kx points where
                                                    % interpolation is done, and length(KX) = nx
                                                    % kx: unevenly spaced array
kxc=KX(nx/2+1);                          % carrier frequency in kx domain
KX=KX(:)*ones(1,ny);                     % nx x ny grid for kx
%
F=zeros(nx,ny);         % initialize F(kx,ky) array for interpolation

for i=1:n                      % for each k loop
                                 % print i to show that it is running
     icKX=round((kx(i,:)-KX(1,1))/dkx)+1; % closest grid point in KX domain % columns of KX are the same 
                                                            % i: fast-time sampe index
     cKX=KX(1,1)+(icKX-1)*dkx;            % and its KX value
     ikx=ones(I,1)*icKX+[-is:is]'*ones(1,ny); % l x ny
     ikx=ikx+nx*ones(I,1)*[0:ny-1];
     nKX=KX(ikx); % (l=17) x ny 
     SINC=sinc((nKX-ones(I,1)*kx(i,:))/dkx);             % interpolating sinc
     HAM=.54+.46*cos((pi/kxs)*(nKX-ones(I,1)*kx(i,:)));  % Hamming window % kxs = Ns*dkx, l=17
         %%%%%   Sinc Convolution (interpolation) follows  %%%%%%%%
     F(ikx)=F(ikx)+(ones(I,1)*fsm(i,:)).*(SINC.*HAM); % F is the interpolated F(kx, ky)
end


%
%  DISPLAY interpolated spatial frequency domain image F(kx,ky)

KX=KX(:,1).';
KY=ky(1,:);

[F,KX,KY,kxc] = interpolation_code(fsm,kx,ky,dkx,nx,ny,kxmin,n); % F is the interpolated spectrum
 

G=abs(F)'; % G is what you need to plot (after mapping it to the range [1, 256])
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(7)   %P4.7
colormap(gray(256));
image(KX,KY+kus(1),256-cg*(G-ng)); % x''=256-x'
axis('square');axis('xy')
xlabel('Spatial frequency k_x, rad/m');
ylabel('Spatial frequency k_y, rad/m'); % F(kx,ky): spectrum
title('Spatial frequency interpolation reconstruction spectrum'); %DS=digital spotlighting
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f=iftx(ifty(F));
dx=pi2/(nx*dkx);
x=dx*(-nx/2:nx/2-1);

G=abs(f)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);

figure(8)   %P4.8
colormap(gray(256));
image(Xc+x,Yc+y,256-cg*(G-ng)); 
axis([Xc-X0 Xc+X0 Yc-Y0 Yc+Y0]);
axis image; axis xy
xlabel('Range x,meters');
ylabel('Cross range y,meters');
title('Spatial frequency interpolation reconstruction'); %DS=digital spotlighting
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%% range stack reconstruction

f_stack=zeros(nx,ny);

for i=1:nx
    
    f_stack(i,:)=ifty(sum(fs.*exp(cj*kx*(Xc+x(i))+cj*ky*Yc-cj*2*k(:)*ones(1,ny)*Rc)));
    
end

f_stack=f_stack.*exp(-cj*x(:)*kxc*ones(1,ny));

f_stack=f_stack/nx;

G=abs(f_stack)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(9) %P4.9
colormap(gray(256));
image(Xc+x, Yc+y, 256-cg*(G-ng));
axis([Xc-X0 Xc+X0 Yc-Y0 Yc+Y0]);
axis image;axis xy
xlabel('Range x, meters');
ylabel('Cross Range y, meters');
title('Range Stack Spotlight SAR Reconstruction'); 
grid on
set(gca,'fontsize',16);


F_stack= ftx(fty(f_stack));

G=abs(F_stack)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);

figure(10) %P4.10
colormap(gray(256));
image(KX, KY+kus(1), 256-cg*(G-ng));
axis([Xc-X0 Xc+X0 Yc-Y0 Yc+Y0]);
axis image;axis xy
xlabel('Spatial frequency k_x, rad/m');
ylabel('Spatial frequency k_y, rad/m');
title('Range Stack Spotlight SAR Reconstruction Spectrum'); 
grid on
set(gca,'fontsize',16);

%%%%%%%%%%%%%%%%%%%Backprojection Reconstruction

f_back=zeros(nx,ny);
n_ratio=100;
nu=n_ratio*n;
nz=nu-n;
dtu=(n/nu)*dt;
tu=dtu*(-nu/2:nu/2-1);
X=x(:)*ones(1,ny);
Y=ones(nx,1)*y;

for j=1:m
    
    t_ij=(2*sqrt((X+Xc).^2+(Y+Yc-u(j)).^2))/c;
    t_ij=round((t_ij-tm(n/2+1))/dtu)+nu/2+1;
    S=ifty([zeros(1,nz/2),s_ds(:,j).',zeros(1,nz/2)])...
        .*exp(cj*wc*tu);
    f_back=f_back+S(t_ij);
    
end

clear X Y

f_back=f_back.*exp(-cj*x(:)*kxc*ones(1,ny));

f_back=f_back.*exp(-cj*ones(nx,1)*2*kc*sin(theta_c)*y);

G=abs(f_back)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(11) %P4.11
colormap(gray(256));
image(Xc+x, Yc+y, 256-cg*(G-ng));
axis([Xc-X0 Xc+X0 Yc-Y0 Yc+Y0]);
axis image;axis xy
xlabel('Range x, meters');
ylabel('Cross Range y, meters');
title('Backprojection Spotlight SAR Reconstruction'); 
grid on
set(gca,'fontsize',16);

F_back=ftx(fty(f_back));

G=abs(F_back)';
xg=max(max(G)); ng=min(min(G)); cg=255/(xg-ng);


figure(12) %P4.12
colormap(gray(256));
image(KX,KY+kus(1),256-cg*(G-ng)); % x''=256-x'
axis([Xc-X0 Xc+X0 Yc-Y0 Yc+Y0]);
axis image;axis xy
xlabel('Spatial frequency k_x, rad/m');
ylabel('Spatial frequency k_y, rad/m');
title('Backprojection Spotlight SAR Reconstruction spectrum'); 
grid on
set(gca,'fontsize',16);



















