function f = h_lambda_fun(y,mu,sigma2,alpha_hp,beta_hp,mu_hp,sigma2_hp,mu_mu,sigma2_mu,alpha_sigma2,beta_sigma2)
n = length(y);

log_p_mu = -1/2*log(2*pi)-1/2*log(sigma2_hp)-(mu-mu_hp)^2/2/sigma2_hp;
log_p_sigma2 = alpha_hp*log(beta_hp)-gammaln(alpha_hp)-(alpha_hp+1)*log(sigma2)-beta_hp/sigma2;
log_q_mu = -1/2*log(2*pi)-1/2*log(sigma2_mu)-(mu-mu_mu)^2/2/sigma2_mu;
log_q_sigma2 = alpha_sigma2*log(beta_sigma2)-gammaln(alpha_sigma2)-(alpha_sigma2+1)*log(sigma2)-beta_sigma2/sigma2;
llh = -n/2*log(2*pi)-n/2*log(sigma2)-1/2/sigma2*sum((y-mu).^2);

f = log_p_mu+log_p_sigma2+llh-log_q_mu-log_q_sigma2;

end


