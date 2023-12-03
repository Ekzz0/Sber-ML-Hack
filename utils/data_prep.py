import pandas as pd
import numpy as np


def check_cat(df, cat_name, col_name):
    df = df[df[col_name] == cat_name]['gender'].value_counts()
    return df


def get_cats_to_change(df_old, col_name='mcc_description', threshold=0.1):
    info = []
    cats_to_change = []
    for cat in df_old[col_name].unique():
        df = check_cat(df_old, cat, col_name)
        cats = {}
        try:
            gen_0 = df[df.index == 0].values[0]
            gen_1 = df[df.index == 1].values[0]
        except IndexError:
            # Если данный товар покупали только мужчины или только женщины:
            cats['cat'] = cat
            cats['gen'] = df.index.values[0]
            if df.index.values[0] == 1:
                per_1 = 1.0
                gen_1 = 1.0
                per_0 = 0.0
                gen_0 = 0.0
            else:
                per_1 = 0.0
                gen_1 = 0.0
                per_0 = 1.0
                gen_0 = 1.0
            cats_to_change.append(cats)
            info.append(
                {'cat_name': cat, 'per_0': per_0, 'per_1': per_1, 'diff': abs(per_0 - per_1), '0': gen_0, '1': gen_1, })
        else:
            # Если данный товар покупали оба представителя разного пола
            per_0 = gen_0 / (gen_1 + gen_0)
            per_1 = gen_1 / (gen_1 + gen_0)
            diff = abs(per_0 - per_1)

            if diff > threshold:
                cats['cat'] = cat
                cats['gen'] = 0 if (per_0 - per_1) > 0 else 1
                cats_to_change.append(cats)
                info.append({'cat_name': cat, 'per_0': per_0, 'per_1': per_1, 'diff': abs(per_0 - per_1), '0': gen_0,
                             '1': gen_1, })
    return cats_to_change, info


def create_new_col(df, new_col_name, to_change, col_name='mcc_description'):
    df[new_col_name] = 0
    for dict_cat in to_change:
        indexes = np.array(df[df[col_name] == dict_cat['cat']].index)
        df.loc[indexes, new_col_name] = dict_cat['gen']
    return df


def create_stereotypical_feature(df, feature_name, stereotypical_list, col_name):
    df[feature_name] = 0
    mask = df[col_name].isin(stereotypical_list)
    df.loc[mask, feature_name] = 1
    return df


def rare_cats_transform(df, col_name, threshold):
    rare = []
    indexes = list(df[col_name].value_counts().index)
    values = list(df[col_name].value_counts().values)

    for info_df in zip(indexes, values):
        if info_df[1] / len(df) <= threshold:
            rare.append(info_df[0])
    df[col_name].mask(df[col_name].isin(rare), 'rare', inplace=True)
    return df


def get_conf_interval(a: list[float], interv: float):
    """
    Вычисление доверительного интервала для значений WoE.
    :param woe: WoE значения.
    :param interv: интервал в долях.
    :return: левая и правая граница доверительного интервала.
    """
    lb = (100-interv*100)/2
    rb = interv*100 + lb

    low, right = np.percentile(a=a, q=[lb, rb])
    return low, right

