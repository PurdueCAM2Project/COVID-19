from scipy.stats import shapiro, mannwhitneyu, ttest_ind
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def check_statistical_similarity(a, b, normal_conf_level=0.005, var_diff_thresh=0.01):
    try:
        assert (np.size(a) >= 3) & (np.size(b) >= 3)
        normal_assertion_true = (shapiro(a)[1] > normal_conf_level) & (
            shapiro(b)[1] > normal_conf_level)
        all_same = (len(set(a)) == 1) & (len(set(b)) == 1)
        if (normal_assertion_true) & (not all_same):
            sigma_a = np.std(a)
            sigma_b = np.std(b)
            not_significant_difference_in_variance = 2 * \
                np.abs(sigma_a-sigma_b)/(sigma_a+sigma_a) <= var_diff_thresh
            p_val = ttest_ind(
                a, b, equal_var=not_significant_difference_in_variance)[1]
        else:
            p_val = mannwhitneyu(a, b)[1]
        return p_val
    except AssertionError:
        print("Not enough data. Returning -1.")
        return -1
