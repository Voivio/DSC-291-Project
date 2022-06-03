## Slide 1

Hello everyone, I am Dongyin Hu and my project title is Multiple testing on experimental economics data. In this project, I applied a FWER control procedure proposed in a paper to an econimics dataset. I also applied this procedure to discover some new insights not mentioned in the papers. Now let us get started.

## Slide 2

Beneﬁted from the improvements in storage, communication and new technologies, the size of the data people have access to nowadays is ever growing. As reported in a work, the medical image research ﬁeld has observed an growth of the geometric mean dataset size in an exponential trend. The researchers are more and more likely to deal with large dataset, and as well conducting many hypothesis tests to discover insights of the data. 

However, recall what we learned from the class, suppose each test is conducted at level α and the p-values are independent, then the family-wise error rate (FWER), the probability of one or more false rejections, equals 1−(1−α) N. This plot shows the growth of FWER w.r.t. the number of hypotheses with $\alpha = 0.05$. We are 22.63% likely to make a false rejection with merely 5 hypotheses, and 99.4% likely to make a false rejection when the size of total testing is N = 100. This is the multiple testing problem. Many methods have been proposed to deal with this problem, such as Bonferroni, and I will be introducing a different control procedure.

## Slide 3

Let us first define the notations. This might seem to be intimadating, but we have to use them to understand the procedure. Assume we have the data $X^{(n)}$ drawn from an unknown underlying distribution $P$ with a sample size $n$. We have $m$ hypotheses to be tested, and the set $S$ is the index set. $I(P)$ is the index set of all real null hypotheses. We denote the null hypotheses as $H_i$ and $i$ is drawn from the index set $S$. To test each null hypotheses, we calculate a test statistics $\hat{T}_{n,i}$ where $n$ is to denote that the statistics is an estimation based on $n$ samples and $i$ is the null hypothesis index. Similarly, we donote the real test statistics under null hypothesis is $T_i(P)$. We reject the null hypothesis if the absolute difference between the estimation and real value is too large. This difference itself is a random variable since estimation is a random variable, and we denote its distribution as $H_{n,i}$. Note this is the distribution function rather than null hypothesis. Then, a $\gamma$ confidence interval can be denoted as below, with the $c_{n,i}(\gamma, P)$ as the largest $\gamma$ quantile of $H_{n, i}$.
$$
\mathbb{P}\{|\hat{T}_{n,i} - T_i| \le c_{n, i}\} = H_{n,i}(c_{n,i}) = \gamma
$$

## Slide 4

Thus we write the FWER in the following way. Recall that $I(P)$ is the index set of all real null hypotheses, 1 - FWER equals to the probability of rejecting at most 1 real null hypothese, and this is equal to the second line, as we discussed in the previous slide. This is equal to the maximum of ~ is smaller than $\gamma$. Recall that taking maximum of r.v.s results a r.v., so this giant term is a random variable. If $\gamma$ is larger than the $1-\alpha$ largest quantile of this random variable, we can continue to the last line where this probability is smaller than $1-\alpha$. 

This looks good. If we know $\gamma$, we can get the confidence interval for all test statistics, and conclude the results with FWER control. But the problem is: how do we get this distribution? A further question, how do we get the distribution of each test statistic so as to calculate the maximum over them?

## Slide 5

To estimate the distribution of the two r.v.s, the proposed procedure in the paper is to use bootstrap. The Bootstrap method is to repetitively draw samples from the existing samples to estimate the property of a test statistics, such as variance or ecdf in our case. The procedure is simple, we randomly sample $n$ data with replacement from the ecisting dataset, and calculate some statistic, and repeat $B$, for example, 3000 times. Then we get a series of the test statistics and it is now possible to calculate the cariance, or more complicated, the ecdf.

In our case, in order to get this ecdf, we sample $n$ data with replacement for $B$ times, and this gives us $B$ this absolute difference. Then we can estimate the ecdf. Further, we repeat the bootstrap procedure for $|S|$ hypotheses, and take the maximum over the hypotheses index, and this results $B$ values for estimation of the second distribution.

[Potentially use table to illustrate]

## Slide 6

This slide serves as a quick recap of the dataset. The dataset is collected to investigate the effect of matching grants, or matching gifts. In the presence of matching grant, a leadership donor promises to offer an extra amount of money to your gift until a predetermined threshold is reached. For example, when you donate \$1, the donor will donate \$2, if the matching ratio is set as 2:1. And the total extra amount is set to a certain limit which can be \$25,000 for example. The donor will not add more money if \$25,000 has been given out.

In the experiment, they sent out letters to solicite contributions from previous donors. Matching ratios, matching threshold, and the suggested amount of donation are all different treatments, as highlighted in red, orange and blue. 3 ratios, 4 thresholds, and 3 levels of recommendation are tested over 50,083 subjects. Four different outcomes are collected, including reponse or not, dollars donated, dollars donated with matching amount, and donation amount change. Based on the states and counties the subject lives in, the samples can be further divied into 4 categories that are red / blue.

The test statistics used here is as the equation shows. This is just the absolute difference of means of different groups. Each group can be divided based on treatment or subgroups. The statistic is centralized, but not standardized. To achiece higher order asympototic property, one can definitely do this.

## Slide 7

Cool. So much for the boring math we can finally see some conclusions. First I reproduce the results in the paper proposed this procedure to check my results, since I implement the same algorithm in Python, which is more suitable for my workflow.

The results are almost similar to the original results, except for 1) the way I calculate the $p$-values, and 2) the results wr.t. the blue states as in Panel B. Even the sbsolute difference does not match with the original results. I think this is a problem with the dataset itself, though if we only check the statistical significance, the conclusions are still the same.

This table summarizes the results for 3 cases: examing multiple outcomes, subgroups, and treatments. I highlighted the significant results. The first column shows the absolute difference between 2 groups, and the later four columns are $p$-values from 1) not adjusted results, 2) adjusted using the previous mentioned method, 3) Bonferroni method, and 4) Holm's methods. The later is a updated version of Bonferroni and is more powerful.

Without adjustment, one would conclude that the treatment, or the matching grant is effective for blue county in red states, and the 2:1 seems to be most effective w.r.t. other matching ratios. Yet after adjustment, it is not significant for these 2 conclusion. Also note the result on dollars given without matching. This procedure rejects it at 10% level while Bon and Hold do not, which shows the proposed procedure is more powerful than these two.

It seems that matching grant does affect the reponse rate and dollar donated. It is more powerful in red states, and the differnt matching ratios are actually have similar effect, which is counterintuitive.

## Slide 8

Now let's look at some new insights. I conducted the similar procedure to study the influence of matching grant on reponse rate and dollar donated for 1) different gender 2) median household income levels and 3) average household size levels, as suggested in the previous presentation. High/low MHI is decided on whether the MHI of a sample is greater of less than the median. The results shows that matching grants are more effective on male, families with relatively lower median household income and relatively small household size on reponse rate. While the amount donated is not signicant after adjustment.

## Slide 9

In addition, I used some regression models to analyze the data.  First we try to find out the relationship between an indicator variable for responding or not w.r.t. the treatments. The model used here is Probit, a generalized linear model which fits the probability distribution of a binary variable as a function of a linear combination of independent variables.

The numbers in the table are the marginal effect of each variable. Marginal effects tells us how a dependent variable (outcome) changes when a specific independent variable (explanatory variable) changes, which can be understood as a derivative.

Similar as previous analysis, the treatment is effective on male, and families with low MHI, small AHS. Also notice 3:1 ratio seems to be significant on small AHS.

## Slide 10

This table shows the ordinary least squares LR results. We regress the amount of dollar given w.r.t. the treatments. Similar as the unadjusted results, the treatment influences the male and families with low MHI. But recall that the correction procedure actually do not reject these null hypotheses.

## Slide 11

This slide shows a summary of the project.