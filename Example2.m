% Implement FFVB with control variate and adaptive learning
% for the normal model example (Example 3.1)
% Implement the MFVB for the normal model example
% Reference: "A practical tutorial on Variational Bayes" by Tran, Nguyen and Dao 
%% =================== FFVB =========================%
    clear all;  
    clc
    rng(1000)
    n = 10;
    y = [11; 12; 8; 10; 9; 8; 9; 10; 13; 7]; % data
    %===========================
    d = 4;
    S = 1000;  % number of Monte Carlo samples 
    beta1_adap_weight = 0.9; % adaptive learning weight
    beta2_adap_weight = 0.9; % adaptive learning weight
    eps0 = 0.1; 
    w_adadelta = 0.95; % adaptive learning weight
    eps_adadelta = 1e-7; % adaptive learning eps
    
    max_iter = 2000;
    patience_max = 20;
    tau_threshold = max_iter/2;
    t_w = 50;

%     optimizer = 'ADADELTA';
    optimizer = 'ADAM';

    % hyperparameter
    alpha_hp = 1; beta_hp = 1; mu_hp = 0; sigma2_hp = 10; 

    lambda = [mean(y);1.5;2;3]; % initial lambda; 
    lambda_best = lambda;

    mu_mu = lambda(1); sigma2_mu = lambda(2); alpha_sigma2 = lambda(3); beta_sigma2 = lambda(4);
    h_lambda = zeros(S,1); % function h_lambda
    grad_log_q_lambda = zeros(S,d);
    grad_log_q_times_h = zeros(S,d);
    parfor s = 1:S    
        % generate theta_s
        mu = normrnd(mu_mu,sqrt(sigma2_mu),1);
        sigma2 = 1./gamrnd(alpha_sigma2,1/beta_sigma2,1);

        grad_log_q_lambda(s,:)=[(mu-mu_mu)/sigma2_mu;-1/2/sigma2_mu+(mu-mu_mu)^2/2/sigma2_mu^2;...
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
    
    switch optimizer
    case 'ADADELTA'
        delta_lambda = grad_LB;
        delta2_bar = zeros(d,1); 
        g2_bar = zeros(d,1);
    case 'ADAM'
        g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
        g_bar_adaptive = g_adaptive; v_bar_adaptive = v_adaptive; 
    end

    iter = 1;
    stop = false;
    LB = 0; LB_bar = 0; patience = 0;
    while ~stop    

        mu_mu = lambda(1); sigma2_mu = lambda(2); alpha_sigma2 = lambda(3); beta_sigma2 = lambda(4);
        h_lambda = zeros(S,1); % function h_lambda
        grad_log_q_lambda = zeros(S,d);
        grad_log_q_times_h = zeros(S,d);
        grad_log_q_times_h_cv = zeros(S,d);
        for s = 1:S    
            % generate theta_s
            mu = normrnd(mu_mu,sqrt(sigma2_mu),1);
            sigma2 = 1./gamrnd(alpha_sigma2,1/beta_sigma2,1);

            grad_log_q_lambda(s,:)=[(mu-mu_mu)/sigma2_mu;-1/2/sigma2_mu+(mu-mu_mu)^2/2/sigma2_mu^2;...
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

        switch optimizer
            case 'ADADELTA'
                delta2_bar_previous = delta2_bar;
                g2_bar = w_adadelta*g2_bar+(1-w_adadelta)*grad_LB.^2;
                rho = sqrt(delta2_bar_previous+eps_adadelta)./sqrt(g2_bar+eps_adadelta);
                delta_lambda = rho.*grad_LB;
                delta2_bar = w_adadelta*delta2_bar+(1-w_adadelta)*delta_lambda.^2;

                lambda = lambda+delta_lambda;
        case 'ADAM'
            g_adaptive = grad_LB; v_adaptive = g_adaptive.^2; 
            g_bar_adaptive = beta1_adap_weight*g_bar_adaptive+(1-beta1_adap_weight)*g_adaptive;
            v_bar_adaptive = beta2_adap_weight*v_bar_adaptive+(1-beta2_adap_weight)*v_adaptive;

            if iter>=tau_threshold
                stepsize = eps0*tau_threshold/iter;
            else
                stepsize = eps0;
            end

            lambda = lambda+stepsize*g_bar_adaptive./sqrt(v_bar_adaptive);
        end

        LB(iter) = mean(h_lambda);

        if iter>=t_w
            LB_bar(iter-t_w+1) = mean(LB(iter-t_w+1:iter));
            LB_bar(iter-t_w+1)
        end

        if (iter>t_w)
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

%% Run Gibbs sampling
% alpha_hp = 1; beta_hp = 1; mu_hp = 0; sigma2_hp = 10; % hyperparameters

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
        scale = 1/(1/sigma2_hp+n/sigma2_mcmc(i));
        location = n*y_bar/sigma2_mcmc(i)*scale;
        mu_mcmc(i+1) = normrnd(location,sqrt(scale));
        aux = gamrnd(n/2+alpha_hp,1/(beta_hp+sum((y-mu_mcmc(i+1)).^2)/2));
        sigma2_mcmc(i+1) = 1/aux;
        i = i+1;
    end
    CPU_MCMC = toc % CPU time taken to run the Gibbs sampling 
    mu_mcmc = mu_mcmc(Nburn+1:N);
    sigma2_mcmc = sigma2_mcmc(Nburn+1:N);

%% Plot the marginal posterior densities     

    fontsize = 20;
    x = 7:.001:13;
    yy_MCMC = ksdensity(mu_mcmc,x,'kernel','normal','function','pdf','width',.14);
    yy_VB = normpdf(x,mu_mu,sqrt(sigma2_mu));
    subplot(1,3,1)
    plot(x,yy_MCMC,'--',x,yy_VB,'-','LineWidth',2);
    xlabel('\mu','FontSize', fontsize)
    legend('MCMC','FFVB')
    set(gca,'FontSize',15)

    x = 0:.0001:10;
    yy_MCMC = ksdensity(sigma2_mcmc,x,'kernel','normal','function','pdf','width',.14);
    inverse_gamma_pdf = @(x) exp(alpha_sigma2*log(beta_sigma2)-gammaln(alpha_sigma2)-(alpha_sigma2+1)*log(x)-beta_sigma2./x);
    yy_VB = inverse_gamma_pdf(x);
    subplot(1,3,2)
    plot(x,yy_MCMC,'--',x,yy_VB,'-','LineWidth',2);
    legend('MCMC','FFVB')
    xlabel('\sigma^2','FontSize', fontsize)
    set(gca,'FontSize',15)

    subplot(1,3,3)
    plot(LB_bar,'LineWidth',2)
    xlabel('LB','FontSize', fontsize)
    set(gca,'FontSize',15)
