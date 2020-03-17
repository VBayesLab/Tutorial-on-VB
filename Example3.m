% Implement FFVB with control variate and natural gradient
% for the normal model example
% Implement the MFVB for the normal model example
% Reference: "A practical tutorial on Variational Bayes" by Minh-Ngoc Tran 

clear all
rng(1000)
n = 10;
y = [11; 12; 8; 10; 9; 8; 9; 10; 13; 7]; %data
%===========================
d = 4;
S = 2000;
momentum_weight = 0.9;
eps0 = 0.001; 
max_iter = 2000;
patience_max = 10;
tau_threshold = max_iter/2;
t_w = 50;

% hyperparameter
alpha_hp = 1; beta_hp = 1; mu_hp = 0; sigma2_hp = 10; 

lambda = [mean(y);.5;1;1]; % initial lambda
lambda_best = lambda;

mu_mu = lambda(1); sigma2_mu = lambda(2); alpha_sigma2 = lambda(3); beta_sigma2 = lambda(4);
h_lambda = zeros(S,1); % function h_lambda
grad_log_q_lambda = zeros(S,d);
grad_log_q_times_h = zeros(S,d);
parfor s = 1:S    
    % generate theta_s
    mu = normrnd(mu_mu,sqrt(sigma2_mu),1);
    sigma2 = 1./gamrnd(alpha_sigma2,1/beta_sigma2,1);
    
    grad_log_q_lambda(s,:)=[-(mu-mu_mu)/sigma2_mu;-1/2/sigma2_mu+(mu-mu_mu)^2/2/sigma2_mu^2;...
        log(beta_sigma2)-psi(alpha_sigma2)-log(sigma2);alpha_sigma2/beta_sigma2-1/sigma2]';
    h_lambda(s) = h_lambda_fun(y,mu,sigma2,alpha_hp,beta_hp,mu_hp,sigma2_hp,mu_mu,sigma2_mu,alpha_sigma2,beta_sigma2);      
    grad_log_q_times_h(s,:) = grad_log_q_lambda(s,:)*h_lambda(s);
end
cv = zeros(1,d); % control variate 
for i = 1:d
    aa = cov(grad_log_q_times_h(:,i),grad_log_q_lambda(:,i));
    cv(i) = aa(1,2)/aa(2,2);
end
grad_LB = mean(grad_log_q_times_h)';
I_igam = [psi(1,alpha_sigma2) -1/beta_sigma2;-1/beta_sigma2 alpha_sigma2/beta_sigma2^2]; % inverse Fisher matrix
grad_LB_nat = [sigma2_mu*grad_LB(1);2*sigma2_mu^2*grad_LB(2);I_igam\grad_LB(3:4)];
grad_LB_bar = grad_LB_nat;

iter = 1;
stop = false;
LB = 0; LB_bar = 0; patience = 0;
while ~stop    
    
    mu_mu = lambda(1); sigma2_mu = lambda(2); alpha_sigma2 = lambda(3); beta_sigma2 = lambda(4);
    h_lambda = zeros(S,1); % function h_lambda
    grad_log_q_lambda = zeros(S,d);
    grad_log_q_times_h = zeros(S,d);
    grad_log_q_times_h_cv = zeros(S,d);
    parfor s = 1:S    
        % generate theta_s
        mu = normrnd(mu_mu,sqrt(sigma2_mu),1);
        sigma2 = 1./gamrnd(alpha_sigma2,1/beta_sigma2,1);

        grad_log_q_lambda(s,:)=[-(mu-mu_mu)/sigma2_mu;-1/2/sigma2_mu+(mu-mu_mu)^2/2/sigma2_mu^2;...
            log(beta_sigma2)-psi(alpha_sigma2)-log(sigma2);alpha_sigma2/beta_sigma2-1/sigma2]';
        h_lambda(s) = h_lambda_fun(y,mu,sigma2,alpha_hp,beta_hp,mu_hp,sigma2_hp,mu_mu,sigma2_mu,alpha_sigma2,beta_sigma2);    
        grad_log_q_times_h(s,:) = grad_log_q_lambda(s,:)*h_lambda(s);
        grad_log_q_times_h_cv(s,:) = grad_log_q_lambda(s,:).*(h_lambda(s)-cv);
    end
    cv = zeros(1,d); % control variate 
    for i = 1:d
        aa = cov(grad_log_q_times_h(:,i),grad_log_q_lambda(:,i));
        cv(i) = aa(1,2)/aa(2,2);
    end
    grad_LB = mean(grad_log_q_times_h_cv)';
 
    I_igam = [psi(1,alpha_sigma2) -1/beta_sigma2;-1/beta_sigma2 alpha_sigma2/beta_sigma2^2];
    grad_LB_nat = [sigma2_mu*grad_LB(1);2*sigma2_mu^2*grad_LB(2);I_igam\grad_LB(3:4)];

    grad_LB_bar = momentum_weight*grad_LB_bar+(1-momentum_weight)*grad_LB_nat;

    if iter>=tau_threshold
        stepsize = eps0*tau_threshold/iter;
    else
        stepsize = eps0;
    end
    
    lambda = lambda+stepsize*grad_LB_bar;
    
    LB(iter) = mean(h_lambda);
    
    if iter>=t_w
        LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
        LB_bar(iter-t_w+1)
    end
       
    if iter>t_w
        if (LB_bar(iter-t_w+1)>=max(LB_bar))
            lambda_best = lambda;
            patience = 0;
        else
            patience = patience+1;
        end
    end
    
    if (patience>patience_max)||(iter>max_iter) stop = true; end 
        
    iter = iter+1;
 
end
lambda = lambda_best;
mu_mu = lambda(1); sigma2_mu = lambda(2); alpha_sigma2 = lambda(3); beta_sigma2 = lambda(4);

    
fontsize = 20;
x = 7:.001:13;
yy_FFVB = normpdf(x,mu_mu,sqrt(sigma2_mu));
subplot(1,3,1)
plot(x,yy_FFVB,'-','LineWidth',2);
xlabel('\mu','FontSize', fontsize)

x = 0:.001:10;
inverse_gamma_pdf = @(x) exp(alpha_sigma2*log(beta_sigma2)-gammaln(alpha_sigma2)-(alpha_sigma2+1)*log(x)-beta_sigma2./x);
yy_FFVB = inverse_gamma_pdf(x);
subplot(1,3,2)
plot(x,yy_FFVB,'-','LineWidth',2);
xlabel('\sigma^2','FontSize', fontsize)

subplot(1,3,3)
plot(LB_bar,'LineWidth',2)
xlabel('LB','FontSize', fontsize)

