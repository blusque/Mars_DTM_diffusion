betas：$\beta$

alphas：$\alpha = 1-\beta$

alphas_cumprod：$\overline{\alpha_t} = \prod_{s=1}^{t}\alpha_s$

alphas_cumprod_prev:$\overline{\alpha_{t-1}}$

sqrt_recip_alphas: $1/\sqrt{\overline{\alpha_t}}$

sqrt_alphas_cumprod: $\sqrt{\overline{\alpha_t}}$

sqrt_one_minus_alphas_cumprod: $\sqrt{1-\overline{\alpha_t}}$

posterior_variance: $\beta * (1-\overline{\alpha_{t-1}}) / (1-\overline{\alpha_{t}})$