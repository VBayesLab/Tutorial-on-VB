% Implement the MFVB for the normal model example
% Reference: "A practical tutorial on Variational Bayes" by Minh-Ngoc Tran 
clear all
rng(2020)
y = [11; 12; 8; 10; 9; 8; 9; 10; 13; 7];
n = length(y);
alpha0 = 1; beta0 = 1; mu0 = 0; sigma20 = 10; % hyperparameters

% run Gibbs samppling
tic
n = length(y);
Nburn = 10000;
Niter = 20000;
N = Nburn + Niter;
mu_mcmc = zeros(N,1);
sigma2_mcmc = zeros(N,1);
y_bar = mean(y);
i = 1;
mu_mcmc(1) = y_bar; sigma2_mcmc(1) = var(y); %initial value
while i<N
    scale = 1/(1/sigma20+n/sigma2_mcmc(i));
    location = n*y_bar/sigma2_mcmc(i)*scale;
    mu_mcmc(i+1) = normrnd(location,sqrt(scale));
    aux = gamrnd(n/2+alpha0,1/(beta0+sum((y-mu_mcmc(i+1)).^2)/2));
    sigma2_mcmc(i+1) = 1/aux;
    i = i+1;
end
CPU_MCMC = toc % CPU time taken to run the Gibbs sampling 
mu_mcmc = mu_mcmc(Nburn+1:N);
sigma2_mcmc = sigma2_mcmc(Nburn+1:N);

% run MFVB
tic
n = length(y); sum_y2 = sum(y.^2); y_bar = mean(y);
muq = y_bar; sigma2q = 1; % initialise muq and sigma2q
eps = 10e-5;
alphaq = alpha0+n/2;
betaq = beta0+sum_y2/2-n*y_bar*muq+n*(muq^2+sigma2q)/2;
parameter_new = [alpha0,beta0,mu0,sigma20];
stop = 0;
while ~stop
    parameter_old = parameter_new;
    betaq = beta0+sum_y2/2-n*y_bar*muq+n*(muq^2+sigma2q)/2; % update beta_q. No need to update alpha_q
    sigma2q = 1/(1/sigma20+n*alphaq/betaq); % update sigma2_q
    muq = (mu0/sigma20+n*y_bar*alphaq/betaq)*sigma2q; % update mu_q
    parameter_new = [alphaq,betaq,muq,sigma2q];
    if norm(parameter_new-parameter_old)<eps stop = 1; end
end
CPU_VB = toc % CPU time taken to run VB

fontsize = 20;
x = 7:.0001:13;
yy_MCMC = ksdensity(mu_mcmc,x,'kernel','normal','function','pdf','width',.14);
yy_VB = normpdf(x,muq,sqrt(sigma2q));
subplot(1,2,1)
plot(x,yy_MCMC,'--',x,yy_VB,'-','LineWidth',2);
xlabel('\mu','FontSize', fontsize)
legend('MCMC','VB')

x = 0:.0001:10;
yy_MCMC = ksdensity(sigma2_mcmc,x,'kernel','normal','function','pdf','width',.14);
inverse_gamma_pdf = @(x) exp(alphaq*log(betaq)-gammaln(alphaq)-(alphaq+1)*log(x)-betaq./x);
yy_VB = inverse_gamma_pdf(x);
subplot(1,2,2)
plot(x,yy_MCMC,'--',x,yy_VB,'-','LineWidth',2);
legend('MCMC','VB')
xlabel('\sigma^2','FontSize', fontsize)






