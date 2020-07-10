
function BS_1D

close all
clear all

%
% BS parameters
%

S0    = 100;
K     = 100;
r     = 0.05;
sigma = 0.2;
T     = 1;

Smax = 200;
J    = 255;
S    = linspace(0,Smax,J+1)';
J0   = 1 + (J-1)/2;

%
% explicit and implicit solvers
%

figure; 
pos=get(gcf,'pos'); pos(3:4)=pos(3:4).*[0.8 1.2]; set(gcf,'pos',pos);

fprintf('\nValue at strike, and errors in value and first two derivatives\n')

for pass = 1:2
  u = max(0,S-K);   % European call payoff
  if pass==1
    N = 50000;
    u = BS_explicit(r,sigma,T,S,K,u,N);
  else
    N = 2500;
    u = BS_implicit(r,sigma,T,S,K,u,N);
  end

  dS   = Smax/J;
  u0   = 0.5*(u(J0+1)+u(J0));
  up0  =     (u(J0+1)-u(J0)) / dS;
  upp0 = 0.5*(u(J0+2)-u(J0+1)-u(J0)+u(J0-1)) / dS^2;

%
% get errors relative to analytic solution
%

  err1 = u0    - european_call(r,sigma,T,S0,K,'value');
  err2 = up0   - european_call(r,sigma,T,S0,K,'delta');
  err3 = upp0  - european_call(r,sigma,T,S0,K,'gamma');
  if pass==1
    fprintf('explicit: ');
  else
    fprintf('implicit: ');
  end
  fprintf('%18.14f %11.4g %11.4g %11.4g\n',u0, err1, err2, err3);

%
% plot results
%

  subplot(2,1,pass)
  plot(S,u,'rx',S,european_call(r,sigma,T,S,K,'value'),'-b')
  axis([60 140 0 40])
  legend('numerical','analytic','Location','SouthEast')
  if pass==1
    title('explicit solver')
    u0sav = u0;
  else
    title('implicit solver')
  end
end


%-----------------------------------------------------------
%
% implicit Black-Scholes solution
%

function u = BS_implicit(r,sigma,T,S,K,u,N);

dS     = S(2)-S(1);
dt     = T/N;
lambda = 0.5*dt*sigma^2*S.^2/dS^2;
gamma  = 0.5*dt*r*S/dS;

%
% define coefficients of tridiagonal matrix
%

a =   - ( lambda - gamma );
b = 1 + 2*lambda + r*dt;
c =   - ( lambda + gamma );

%
% special treatment of last point, based on d^2u/dS^2=0
%

a(end) =   + 2*gamma(end);
b(end) = 1 - 2*gamma(end) + r*dt;
c(end) = 0;

%
% call function to create sparse tri-diagonal matrix
%

A = sparse_tri(a,b,c);

%
% perform N timesteps
%

for n = 1:N
  u = A\u;
end


%-----------------------------------------------------------
%
% explicit Black-Scholes solution
%

function u = BS_explicit(r,sigma,T,S,K,u,N);

dS     = S(2)-S(1);
dt     = T/N;
lambda = 0.5*dt*sigma^2*S.^2/dS^2;
gamma  = 0.5*dt*r*S/dS;

%
% define coefficients of tridiagonal matrix
%

a =       lambda - gamma;
b = 1 - 2*lambda - r*dt;
c =       lambda + gamma;

%
% special treatment of last point, based on d^2u/dS^2=0
%

a(end) =   - 2*gamma(end);
b(end) = 1 + 2*gamma(end) - r*dt;
c(end) = 0;

%
% call function to create sparse tri-diagonal matrix
%

A = sparse_tri(a,b,c);

%
% perform N timesteps
%

for n = 1:N
  u = A*u;
end


%-----------------------------------------------------------
%
% function to construct tridiagonal matrix
% from 3 vectors defining its diagonals
%

function A = sparse_tri(a,b,c)

A = spdiags([c b a],[-1 0 1],length(a),length(a))';
