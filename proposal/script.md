## Slide 1

Hello everyone, I am Dongyin Hu and my project title is An FWER control procedure on experimental economics data. I am from the ECE department and have only a shallow understanding about the economics problem being discussed here. Please correct me if you notice anything that is confusing.

## Slide 2

In this project, I would like to investigate a problem related with fundraising and charity. An organization may solicite money from subscribers for some reasons, for example, animal protection, improving education in a certain community, or some political events. A key concern for the fundraisers is, how to raise more money? A simple yet difficult question.

In practice, many fundraisers are using a method called matching grants. There is a leadership donor, who promises to offer an extra amount of money to your gift until a predetermined threshold is reached. So for example, when you donate \$1, the donor will donate \$2, if the matching ratio is 2:1. And the total extra amount is set to a certain limit which can be \$25,000 for example. The donor will not add more money if \$25,000 has been given out.

There is a rule of thumbs for fundraisers: the higher the matching ratio, the more money will be received. The problem is, is this correct?

## Slide 3

So, in a 2007 work, several researchers conducted an experiement to see the effect of matching grants. They sent out letters to solicite contributions from previous donors. In addition to the matching ratios we discussed before, more factors are considered as well: the maximum amout of of the matching gift and the suggested donation amount are changed for different donors.  For the matching ratios, they tested 3 ratios. For the maximum amount, they set 4 values, and for the suggested smount, 3 levels are set as well. This results a total of 36 treaments, which is color coded in blue as shown in the sample data.

Also, since the reason for raising money is political, the researchers also divided the donors into 4 subgroups, which are red/blue state and red/blue counties. This is color coded in yellow. Note some other onformation is collected as well, such as marital status, gender or legal status.

Finally, there are four outcomes of interest - the reponse rate, defined as the number of donation received over the total number of letters sent out, dollars given without the matching amount, dollars given with the matching amount, and the changed inthe amount given.

## Slide 4

Now let's look at some statistics about the results. A total number of 50k subjects are asked, and 1/3 of them are assigned to the control group. The 36 treatments are assigned with equal possibility to all in the treatment groups.

And for the outcomes we concern, they are shown in the 4 bar charts w.r.t. the matching ratios. Note that we have 3 different dimensions of treatments, but for now let us focus on the matching ratios. 

From the left to the right are the reponse rates, amount w/o matching per letter, amount with matching per letter, and amount change. And the x-axis is a 1:1 ratio, 2:2 ratio, 3:1 ratio and the control group. We can see that the reponse rate of the three treatment groups seems to be higher, and the amount w/o matching per letter is higher as well. Note this amount is normalized by the size of each group. And for the amount with matching, of course that is higher than the control group. For the amount change, not a clear trend is observed.

## Slide 5

Now let us do some simple testing to see whether the rule of thumbs works. Although we can actually do more analysis w.r.t. different treatments, subroups, and outcomes. We wonder, does the matching ratio influences the amount w/o matching per letter?

There are in total 4 null hypothesis. The first one, $H_0$, is comparing the influece of the matching grant itself. We gather the results for all 36 possible treatments as the treatment group. Also, we are insterested in the matching ratios. So there are three more null hypotheses to be tested: 3 combinations of ratios in the treament groups.

We can apply some simple testing on the data. I choose the Wald tests for 2 means and also the permutatino testing on the absolute difference of mean. The following table summarizes the p-values. 

From these results it seems that a weak evidence against the null hypothesis $H_0$, which means that the treatment is somehow working in soliciting more money. While for the permutation test, it provides a stronger evidence agianst the null hypothesis. The results are consistent with the analysis in the 2007 paper, yet the statistics procedure applied are different.

Also, note that there is no evidence against the rest hypotheses. A quick interpretation is, using matching ratio is somehow helpful but the a higher ratio does not help fundraisers to attract more donations. This is quite conterintuitive and against the rule of thumbs many believe.

But the problem is not completely solved. What if I want to test more? We have 36 treatments, 4 and even more subgroups and 4 outcomes. The problem then becomes a multiple testing problem. For the net step, I will follow a work published in 2019 to control the FWER as we testing more hypothesis. The procedure they proposed offers asymptotic control of the familywise error rate.