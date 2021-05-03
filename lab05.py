import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def first_round():
    """
    :return: list with two values
    >>> out = first_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    """
    return [0.169, "NR"]


def second_round():
    """
    :return: list with three values
    >>> out = second_round()
    >>> isinstance(out, list)
    True
    >>> out[0] < 1
    True
    >>> out[1] is "NR" or out[1] is "R"
    True
    >>> out[2] is "ND" or out[2] is "D"
    True
    """
    return [0.032, "R", "D"]


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def verify_child(heights):
    """
    Returns a series of p-values assessing the missingness
    of child-height columns on father height.

    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> heights = pd.read_csv(fp)
    >>> out = verify_child(heights)
    >>> out['child_50'] < out['child_95']
    True
    >>> out['child_5'] > out['child_50']
    True
    """
    new_df = heights.copy()
    N = 100
    p_values = {}
    for col in heights.columns[2:]:
        # compute the obs value, we first assign a new is_null for every col child_X
        new_df['is_null'] = new_df[col].isnull()
        group_isn = new_df.loc[new_df['is_null'] == True, 'father']
        group_not = new_df.loc[new_df['is_null'] == False, 'father']
        obs = ks_2samp(group_isn, group_not).statistic

        ks_stats = []
        for _ in range(N):
            # shuffle the father column
            shuffled_col = (
                new_df['father'].sample(replace=False, frac=1).reset_index(drop=True)
            )

            # put them in a table
            shuffled = (
                new_df.assign(**{
                    'father': shuffled_col,
                    'is_null': new_df['is_null']
                })
            )

            grps = shuffled.groupby('is_null')['father']
            ks = ks_2samp(grps.get_group(True), grps.get_group(False)).statistic
            ks_stats.append(ks)
        # compute the p-val and append it to the list
        p_val = np.mean(np.array(ks_stats) > obs)
        p_values[col] = p_val
    return pd.Series(p_values)


def missing_data_amounts():
    """
    Returns a list of multiple choice answers.

    :Example:
    >>> set(missing_data_amounts()) <= set(range(1,6))
    True
    """

    return [1,5]


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def cond_single_imputation(new_heights):
    """
    cond_single_imputation takes in a dataframe with columns 
    father and child (with missing values in child) and imputes 
    single-valued mean imputation of the child column, 
    conditional on father. Your function should return a Series.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> df['child'] = df['child_50']
    >>> out = cond_single_imputation(df)
    >>> out.isnull().sum() == 0
    True
    >>> (df.child.std() - out.std()) > 0.5
    True
    """
    def fill_mean(x):
        return x.fillna(x.mean())

    new_heights['father'] = pd.qcut(new_heights['father'], 4)
    return pd.Series(new_heights.groupby('father').transform(fill_mean)['child'])

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def quantitative_distribution(child, N):
    """
    quantitative_distribution that takes in a Series and an integer 
    N > 0, and returns an array of N samples from the distribution of 
    values of the Series as described in the question.
    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = quantitative_distribution(child, 100)
    >>> out.min() >= 56
    True
    >>> out.max() <= 79
    True
    >>> np.isclose(out.mean(), child.mean(), atol=1)
    True
    """
    prob = np.histogram(child.dropna(), bins=10)[0]
    bins = np.histogram(child.dropna(), bins=10)[1]
    choose_bins = np.random.choice(range(10), p=prob / prob.sum(), size=N)
    return np.random.uniform(bins[choose_bins], bins[choose_bins + 1])


def impute_height_quant(child):
    """
    impute_height_quant takes in a Series of child heights 
    with missing values and imputes them using the scheme in
    the question.

    :Example:
    >>> fp = os.path.join('data', 'missing_heights.csv')
    >>> df = pd.read_csv(fp)
    >>> child = df['child_50']
    >>> out = impute_height_quant(child)
    >>> out.isnull().sum() == 0
    True
    >>> np.isclose(out.mean(), child.mean(), atol=0.5)
    True
    """
    child_null = child[child.isnull()]
    child[child_null.index] = quantitative_distribution(child, child_null.shape[0])
    return child


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def answers():
    """
    Returns two lists with your answers
    :return: Two lists: one with your answers to multiple choice questions
    and the second list has 6 websites that satisfy given requirements.
    >>> list1, list2 = answers()
    >>> len(list1)
    4
    >>> len(list2)
    6
    """
    return [1, 2, 1, 1],['https://www.qq.com/robots.txt', 'https://soundcloud.com/robots.txt',
                         'https://www.popads.net/robots.txt', 'https://facebook.com/robots.txt',
                         'https://www.tmall.com/robots.txt', 'https://www.linkedin.com/robots.txt']




# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['first_round', 'second_round'],
    'q02': ['verify_child', 'missing_data_amounts'],
    'q03': ['cond_single_imputation'],
    'q04': ['quantitative_distribution', 'impute_height_quant'],
    'q05': ['answers']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
