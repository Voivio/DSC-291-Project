from typing import Any
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

import scipy.io

import matplotlib.pyplot as plt

from tqdm.notebook import tqdm


class Index:
    def __init__(self, treatment_1: list[tuple], treatment_2: list[tuple], subgroup: list[tuple], outcome: str,
                 title: str) -> None:
        """Index of a null hypothesis. 
        
        Hs : { E[Y_{i, k}(d) - Y_{i, k}(d') | Z = z] = 0 },
        where Y_{i,k}(d) is the k-th outcome of unit i under treatment d, Z is a random variable
        specifies the subgroup. The index s is a tuple (d, d', z, k).

        Args:
            treatment_1 (list[tuple]): One of the two treatments to be compared. (d in above formula). 
            In each tuple, the first element is the field name in DataFrame, and second element is the corresponding value.
            treatment_2 (list[tuple]): One of the two treatments to be compared. (d' in above formula).
            In each tuple, the first element is the field name in DataFrame, and second element is the corresponding value.
            subgroup (list[tuple]): Specifies a subgroup of samples. (z in above formula). 
            In each tuple, the first element is the field name in DataFrame, and second element is the corresponding subgroup value.
            outcome (str): Specifies which outcome to test. (k in above formula).
            title (str): To present in the final output.
        """
        self.treatment_1 = treatment_1
        self.treatment_2 = treatment_2
        self.subgroup = subgroup
        self.outcome = outcome
        self.title = title


class MHT:
    def __init__(self, B=3000, multiprocess=False) -> None:
        """The multiple hypotheses testing procedure as proposed in 
        Multiple Hypothesis Testing in Experimental Economics. List, Shaikh, and Xu (2015).

        Args:
            B (int, optional): The booststrap simulation times. Defaults to 3000.
            multiprocess (bool, optional): Whether to use multiprocessing ot not in bootstrap. Defaults to False.
        """
        self.B = B
        self.multiprocess = multiprocess

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.test(*args, **kwds)

    def test(self, df, S: list[Index]):
        """Test multiple hypotheses as specified in S with a balanced, asymptotic control of FWER.

        See Multiple Hypothesis Testing in Experimental Economics. List, Shaikh, and Xu (2015)

        Args:
            df: Pandas DataFrame containing all original data.
            S (list(Index)): The index set.
        """
        treatments, subgroups, outcomes, pairs, titles, titles_order = self.index_set_to_conditions(S)
        # print(titles_order)
        fields = self._get_necessary_fields(treatments, subgroups, outcomes)
        df = df[fields]

        self.n_samples = df.shape[0]
        self.n_pairs = len(pairs)
        self.n_treatments = len(treatments)
        self.n_subgroups = len(subgroups)
        self.n_outcomes = len(outcomes)

        print("Different treatments: {}, different subgroups : {}, different outcomes {}, compare pairs: {}".format(
            self.n_treatments, self.n_subgroups, self.n_outcomes, self.n_pairs
        ))

        # =========================================
        # compute the studentized differences in means for all the hypotheses based on the actual data

        # stats : (n_pairs, n_subgroups, n_outcomes)
        diff, stats = self.calculate_statistics(df, treatments, subgroups, outcomes, pairs)
        # bootstrap_stats : (B, n_pairs, n_subgroups, n_outcomes)
        # used to estimate J(x) and L_S(x)
        bootstrap_stats = self.bootstrap(df, treatments, subgroups, outcomes, pairs, diff)

        # calculate individual p-values. p-value is the probability of observing more extreme statistics.
        p_actual = (bootstrap_stats >= stats).mean(axis=0)
        p_boot = np.zeros_like(bootstrap_stats)
        for b in range(self.B):
            p_boot[b, :, :, :] = (bootstrap_stats >= bootstrap_stats[b, :, :, :]).mean(axis=0)

        # =========================================
        # p-value without adjusting.
        # Individual hypothesis is rejected if T > J^{-1}(1-a), or J(T) > 1-a.
        # For calculating p-value, we count how many p-values in bootstrap tests are larger than observaed p-value.
        # Note that J(observation) = 1 - p. As mentioned, we can replace T^ to -p^
        p_single = ((1 - p_boot) >= (1 - p_actual)).mean(axis=0).ravel()

        # ========================================= 
        # calculate p-values based on multiple hypothesis testing
        # For each iteration, a hypotheses if rejected of T > L^{-1}(1-a), or L(T) > 1-a.
        reshape_p_actual = p_actual.reshape(-1, )
        sort_index = np.argsort(reshape_p_actual)
        reshape_p_actual = reshape_p_actual[sort_index]
        reshaped_p_boot = p_boot.reshape(self.B, -1).T[sort_index, :]

        p_adjusted = np.zeros_like(reshape_p_actual)
        for i in range(len(S)):
            # L_S(x) is the distribution of Pr[ max_{s in S} J(T_s) <= x ]
            #   = Pr[ max_{s in S} (1 - p_s) <= x ]
            # L_S : (B,)
            L_S = (1 - reshaped_p_boot[i:]).max(axis=0)

            # p_adj = Pr[ max_{s in S} J(T_s) >= J(T_s) ]
            p_adjusted[i] = (L_S >= (1 - reshape_p_actual[i])).mean()

        # =========================================
        # p-values based on the Bonferroni method
        p_bonferroni = np.minimum(reshape_p_actual * len(S), 1)

        # =========================================
        # p-values based on the Holm's method
        p_holm = np.minimum(reshape_p_actual * np.arange(len(S), 0, -1), 1)

        # =========================================
        # construct output
        reverse_index = np.argsort(np.arange(len(reshape_p_actual))[sort_index])
        output = pd.DataFrame({
            "Hypotheses": titles,
            "Absolute Difference": np.abs(diff).reshape(-1, ),
            "p (unadjusted)": p_single,
            "p (adjusted)": p_adjusted[reverse_index],
            "p (Bonferroni)": p_bonferroni[reverse_index],
            "p (Holm's)": p_holm[reverse_index]
        })

        # =========================================
        # check the results
        check = np.all(
            (output["p (unadjusted)"] <= output["p (adjusted)"]) &
            (output["p (adjusted)"] <= output["p (Bonferroni)"]) &
            (output["p (adjusted)"] <= output["p (Holm's)"])
        )
        print("Check {}.".format("passed" if check else "failed"))

        return output.iloc[titles_order].reset_index(drop=True)

    def calculate_statistics(self, df, treatments, subgroups, outcomes, pairs, diff_central=None):
        n_samples = self.n_samples
        n_treatments = self.n_treatments
        n_subgroups = self.n_subgroups
        n_outcomes = self.n_outcomes

        mean = np.zeros((n_treatments, n_subgroups, n_outcomes))
        var = np.zeros_like(mean)
        n = np.zeros_like(mean)
        for t in range(n_treatments):
            for s in range(n_subgroups):
                for o in range(n_outcomes):
                    condition = np.ones((n_samples,))
                    for field, value in treatments[t]:
                        condition = condition & (df[field] == value)
                    for field, value in subgroups[s]:
                        condition = condition & (df[field] == value)

                    sub_df = df.loc[condition, outcomes[o]]

                    mean[t, s, o] = sub_df.mean()
                    var[t, s, o] = sub_df.var()
                    n[t, s, o] = sub_df.shape[0]

        diff = mean[pairs[:, 0], :, :] - mean[pairs[:, 1], :, :]
        if diff_central is None:
            abs_diff = np.abs(diff)
        else:
            abs_diff = np.abs(diff - diff_central)

        stats = abs_diff / np.sqrt(
            var[pairs[:, 0], :, :] / n[pairs[:, 0], :, :] +
            var[pairs[:, 1], :, :] / n[pairs[:, 1], :, :]
        )

        return diff, stats

    def bootstrap(self, df, treatments, subgroups, outcomes, pairs, diff_central):
        bootstrap_id = np.random.random_integers(0, self.n_samples - 1, (self.B, self.n_samples))
        bootstrap_stats = np.zeros((self.B, len(pairs), self.n_subgroups, self.n_outcomes))

        if self.multiprocess:
            results = \
                Parallel(n_jobs=-1, backend='multiprocessing', verbose=1) \
                    (delayed(self.calculate_statistics)(
                        df.iloc[bootstrap_id[b, :]], treatments, subgroups, outcomes, pairs
                    ) for b in range(self.B))

            for b, (_, stats) in enumerate(results):
                bootstrap_stats[b, :, :, :] = stats
        else:
            for b in tqdm(range(self.B)):
                _, bootstrap_stats[b, :, :, :] = self.calculate_statistics(
                    df.iloc[bootstrap_id[b, :]], treatments, subgroups, outcomes, pairs, diff_central
                )

        return bootstrap_stats

    def index_set_to_conditions(self, S):
        treatments = []
        subgroups = []
        outcomes = []

        for s in S:
            treatments.append(s.treatment_1)
            treatments.append(s.treatment_2)
            subgroups.append(s.subgroup)
            outcomes.append(s.outcome)

        treatments = self._remove_duplicates(treatments)
        subgroups = self._remove_duplicates(subgroups)
        outcomes = list(set(outcomes))

        pairs = []
        for s in S:
            new_pair = (
                treatments.index(s.treatment_1),
                treatments.index(s.treatment_2),
            )
            if new_pair not in pairs:
                pairs.append(new_pair)

        titles = np.empty((len(pairs), len(subgroups), len(outcomes)), dtype=object)
        titles_order = np.zeros_like(titles)
        for i, s in enumerate(S):
            new_pair = (
                treatments.index(s.treatment_1),
                treatments.index(s.treatment_2),
            )
            titles[
                pairs.index(new_pair),
                subgroups.index(s.subgroup),
                outcomes.index(s.outcome)
            ] = s.title
            titles_order[
                pairs.index(new_pair),
                subgroups.index(s.subgroup),
                outcomes.index(s.outcome)
            ] = i

        titles = titles.reshape(-1, )
        titles_order = np.argsort(titles_order.reshape(-1))

        return treatments, subgroups, outcomes, np.array(pairs), titles, titles_order

    @staticmethod
    def _get_necessary_fields(treatments, subgroups, outcomes):
        fields = []

        for t_list in treatments:
            for t in t_list:
                fields.append(t[0])

        for s_list in subgroups:
            for s in s_list:
                fields.append(s[0])

        return list(set(fields + outcomes))

    @staticmethod
    def _remove_duplicates(l):
        clean_list = []
        for item in l:
            # item: list of tuples

            same_flag = False
            for c_item in clean_list:
                # c_item: list of tuples
                if len(item) != len(c_item):
                    # lengths are different. Cannot be the same.
                    continue
                else:
                    # lengths are the same. Compare each tuple.
                    all_tuples_same_flag = True
                    for i, j in zip(item, c_item):
                        if i != j:
                            all_tuples_same_flag = False
                            break

                    if all_tuples_same_flag:
                        # all tuples are the same
                        same_flag = True

            if not same_flag:
                clean_list.append(item)

        return clean_list
