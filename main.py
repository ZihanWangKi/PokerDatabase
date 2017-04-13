from collections import defaultdict
import numpy
import sklearn.svm
import get_percentage as gp
import pandas as pd
from statsmodels.formula.api import ols
from scipy import stats
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import dump_svmlight_file
import optunity
import optunity.metrics
from sklearn import decomposition
import operator
import tkinter
import matplotlib.pyplot as plt


def positive(x):
    if x > 0:
        return 1
    return 0


def mask(df, key, value):
    return df[df[key] == value]


def calculate_real_value_ftr(calculator, ser, index=0):
    rank = [ser.loc['Hole1'][0], ser.loc['Hole2'][0], ser.loc['Comm1'][0], ser.loc['Comm2'][0],
            ser.loc['Comm3'][0]]
    suit = [ser.loc['Hole1'][1], ser.loc['Hole2'][1], ser.loc['Comm1'][1], ser.loc['Comm2'][1],
            ser.loc['Comm3'][1]]
    if index == 0:
        return calculator.input_card(rank, suit).calculate_value()
    elif index == 1:
        rank.append(ser.loc['Comm4'][0])
        suit.append(ser.loc['Comm4'][1])
        return calculator.input_card(rank, suit).calculate_value()
    elif index == 2:
        rank.append(ser.loc['Comm4'][0])
        rank.append(ser.loc['Comm5'][0])
        suit.append(ser.loc['Comm4'][1])
        suit.append(ser.loc['Comm5'][1])
        return calculator.input_card(rank, suit).calculate_value()


def get_range(df, lo, hi):
    return df[lo:hi]


def _clean():
    df = pd.read_csv('poker_data.csv')
    df1 = df[
        ['Timestamp', 'Player', '# of Players', 'Position', 'Bankroll', '$ Bet', '$ Won', 'Preflop', 'Flop', 'Turn',
         'River', 'Comm 1', 'Comm 2', 'Comm 3', 'Comm 4', 'Comm 5', 'Hole 1', 'Hole 2', '$ Preflop', '$ Flop.1',
         '$ Turn.1', '$ River.1', 'PreflopV', 'FlopV', 'TurnV', 'RiverV']]
    df1.columns = ['Timestamp', 'Player', 'NumPlayers', 'Position', 'Bankroll', 'Bet', 'Won', 'PreflopAc', 'FlopAc',
                   'TurnAc', 'RiverAc', 'Comm1', 'Comm2', 'Comm3', 'Comm4', 'Comm5', 'Hole1', 'Hole2', 'PreflopBet',
                   'FlopBet', 'TurnBet', 'RiverBet', 'PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom']

    calculator = gp.GetPercentage()
    df2 = df1.apply(lambda x: pd.Series({'FlopVSim': calculate_real_value_ftr(calculator, x, 0),
                                         'TurnVSim': calculate_real_value_ftr(calculator, x, 1),
                                         'RiverVSim': calculate_real_value_ftr(calculator, x, 2),
                                         'PreflopAcB': x.PreflopAc.lower().count('b'),
                                         'PreflopAcR': x.PreflopAc.lower().count('r'),
                                         'PreflopAcC': x.PreflopAc.lower().count('c'),
                                         'PreflopAcK': x.PreflopAc.lower().count('k'),
                                         'FlopAcB': x.FlopAc.lower().count('b'),
                                         'FlopAcR': x.FlopAc.lower().count('r'),
                                         'FlopAcC': x.FlopAc.lower().count('c'),
                                         'FlopAcK': x.FlopAc.lower().count('k'),
                                         'TurnAcB': x.TurnAc.lower().count('b'),
                                         'TurnAcR': x.TurnAc.lower().count('r'),
                                         'TurnAcC': x.TurnAc.lower().count('c'),
                                         'TurnAcK': x.TurnAc.lower().count('k'),
                                         'RiverAcB': x.RiverAc.lower().count('b'),
                                         'RiverAcR': x.RiverAc.lower().count('r'),
                                         'RiverAcC': x.RiverAc.lower().count('c'),
                                         'RiverAcK': x.RiverAc.lower().count('k'),
                                         'IsWin': positive(x.Won)
                                         }), axis=1)
    df3 = pd.concat([df1, df2], axis=1)
    df3.to_csv('data.csv', index=False)


def _player():
    df = pd.read_csv('data.csv')
    player_map = dict()
    # Name--list(16):bankrank Incr, bankrank Initial, bankIncBet, Total games, Total bets, Total b, r, c, k, handValues
    for index, row in df.iterrows():
        s = row['Player']
        if s not in player_map:
            l = [0 for x in range(16)]
            l[0] = row['Bankroll']
            player_map[s] = l
        l = player_map[s]
        l[1] = row['Bankroll'] - l[0]
        l[2] += row['Won']
        l[3] += 1
        l[4] += row['Bet']
        l[5] += row['PreflopAcB'] + row['FlopAcB'] + row['TurnAcB'] + row['RiverAcB']
        l[6] += row['PreflopAcR'] + row['FlopAcR'] + row['TurnAcR'] + row['RiverAcR']
        l[7] += row['PreflopAcC'] + row['FlopAcC'] + row['TurnAcC'] + row['RiverAcC']
        l[8] += row['PreflopAcK'] + row['FlopAcK'] + row['TurnAcK'] + row['RiverAcK']
        l[9] = (l[9] * (l[3] - 1) + row['PreflopVCom']) / l[3]
        l[10] = (l[10] * (l[3] - 1) + row['FlopVCom']) / l[3]
        l[11] = (l[11] * (l[3] - 1) + row['TurnVCom']) / l[3]
        l[12] = (l[12] * (l[3] - 1) + row['RiverVCom']) / l[3]
        l[13] = (l[13] * (l[3] - 1) + row['FlopVSim']) / l[3]
        l[14] = (l[14] * (l[3] - 1) + row['TurnVSim']) / l[3]
        l[15] = (l[15] * (l[3] - 1) + row['RiverVSim']) / l[3]
    lst = []
    avg = [0 for x in range(16)]

    for k, v in player_map.items():
        lst.append([k] + v)
        for i_ in range(v.__len__()):
            avg[i_] += v[i_]
    for i_ in range(avg.__len__()):
        avg[i_] = (avg[i_] + 0.0) / lst.__len__()

    lst.append(['AVERAGE'] + avg)
    labels = ['Player', 'BankIni', 'BankInc', 'BankIncBet', 'TotalGames', 'TotalBet', 'TotalB', 'TotalR', 'TotalC',
              'TotalK', 'PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']
    df1 = pd.DataFrame.from_records(lst, columns=labels)
    df1.to_csv('players.csv', index=False)

    # Average
    for k in range(lst.__len__() - 1):
        for i_ in range(1, lst[k].__len__()):
            lst[k][i_] -= avg[i_ - 1]
    df2 = pd.DataFrame.from_records(lst, columns=labels)
    df2.to_csv('players_avg.csv', index=False)


def optunity_tune_svm_rbf(X_train_all, y_train_all, x_test, y_test):
    def train1(x_train, y_train, x_test, y_test, para):
        model = sklearn.svm.SVC(C=para[0], gamma=10 ** para[1])
        model.fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc

    @optunity.cross_validated(x=X_train_all, y=y_train_all, num_folds=5)
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, logGamma):
        acc = train1(x_train, y_train, x_test, y_test, (C, logGamma))
        # print("train acc: {}".format(acc))
        return acc

    optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=50, C=[0, 10], logGamma=[-5, 0],
                                                  pmap=optunity.pmap)
    df = optunity.call_log2dataframe(info.call_log)
    df = df.sort_values(by='value', ascending=False)[:10]

    df1 = df.apply(lambda x: pd.Series({'True Acc':
                                            train1(X_train_all, y_train_all, x_test, y_test,
                                                   (x.loc['C'], x.loc['logGamma']))
                                        }), axis=1)
    df3 = pd.concat([df, df1], axis=1)
    print(df3)
    return train1(X_train_all, y_train_all, x_test, y_test, (optimal_rbf_pars['C'], optimal_rbf_pars['logGamma']))


def optunity_tune_svr_rbf(X_train_all, y_train_all, x_test, y_test):
    def train1(x_train, y_train, x_test, y_test, para):
        model = sklearn.svm.SVR(C=para[0], epsilon=para[1], gamma=10 ** para[2])
        model.fit(x_train, y_train)

        auc = model.score(x_test, y_test)
        # decision_values = model.decision_function(x_test)
        # auc = optunity.metrics.roc_auc(y_test, decision_values)

        return auc

    @optunity.cross_validated(x=X_train_all, y=y_train_all, num_folds=5)
    def svr_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, epsilon, logGamma):
        acc = train1(x_train, y_train, x_test, y_test, (C, epsilon, logGamma))
        print("train acc: {}".format(acc))
        return acc

    optimal_rbf_pars, info, _ = optunity.maximize(svr_rbf_tuned_auroc, num_evals=50, C=[0, 10], epsilon=[0.05, 0.2],
                                                  logGamma=[-5, 0], pmap=optunity.pmap)
    df = optunity.call_log2dataframe(info.call_log)
    df = df.sort_values(by='value', ascending=False)[:10]

    df1 = df.apply(lambda x: pd.Series({'True Acc':
                                            train1(X_train_all, y_train_all, x_test, y_test,
                                                   (x.loc['C'], x.loc['epsilon'], x.loc['logGamma']))
                                        }), axis=1)
    df3 = pd.concat([df, df1], axis=1)
    print(df3)
    return train1(X_train_all, y_train_all, x_test, y_test, (optimal_rbf_pars['C'], optimal_rbf_pars['epsilon'],
                                                             optimal_rbf_pars['logGamma']))


def tune_player():
    df = pd.read_csv('players.csv')
    df1 = df.loc[df['TotalGames'] >= 5]
    df1['AvgIncBet'] = df1['BankIncBet'] / df1['TotalGames']
    avg = df1.loc[df['Player'] == 'AVERAGE']
    threshold = float(avg.get('AvgIncBet'))
    df1['AvgBet'] = df1['TotalBet'] / df1['TotalGames']
    df1['AvgB'] = df1['TotalB'] / df1['TotalGames']
    df1['AvgR'] = df1['TotalR'] / df1['TotalGames']
    df1['AvgC'] = df1['TotalC'] / df1['TotalGames']
    df1['AvgK'] = df1['TotalK'] / df1['TotalGames']

    # X = df1[['AvgB', 'AvgR', 'AvgC', 'AvgK', 'PreflopVCom', 'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']]
    # X = df1[['AvgR', 'AvgC', 'RiverVCom', 'FlopVSim']]
    res = []
    for predictors in itertools.combinations(
            ['AvgB', 'AvgR', 'AvgC', 'AvgK', 'PreflopVCom', 'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim'], 2):
        print(predictors)
        X = df1[list(predictors)]
        y = df1.apply(lambda x: 1 if x.get('AvgIncBet') > threshold else 0, axis=1)
        X_train = X.ix[:699, :].values
        y_train = y.ix[:699, ].values
        X_test = X.ix[700:, :].values
        y_test = y.ix[700:, ].values
        res.append((predictors, optunity_tune_svm_rbf(X_train, y_train, X_test, y_test)))
    print(sorted(res, key=lambda x: x[1], reverse=True))


def vif_analysis(df, col, threshold):
    intercept = pd.np.ones(df[col[0]].__len__())
    # lis = [df[x] for x in col]
    # design_matrix = pd.np.hstack(lis)
    design_matrix = pd.np.array([df[x] for x in col])
    print(design_matrix)
    design_matrix = pd.np.concatenate((design_matrix.T, intercept))
    print(len(design_matrix[0]))
    while 1:
        vif = [variance_inflation_factor(design_matrix, i) for i in range(design_matrix.shape[1])]
        index, value = max(enumerate(vif), key=operator.itemgetter(1))
        if value < threshold:
            break
        for i in range(len(design_matrix)):
            del design_matrix[i][index]
        print(len(design_matrix[0]))
        del col[index]
    return col


def tune_river_value_com():
    df = pd.read_csv('data.csv')
    all_predictors = ['PreflopBet', 'FlopBet', 'TurnBet', 'RiverBet', 'FlopAcB',
                      'FlopAcC',
                      'FlopAcK', 'FlopAcR', 'PreflopAcB', 'PreflopAcC', 'PreflopAcK', 'PreflopAcR',
                      'RiverAcB',
                      'RiverAcC', 'RiverAcK', 'RiverAcR', 'TurnAcB', 'TurnAcC', 'TurnAcK', 'TurnAcR']
    un_col_predictors = ['PreflopBet', 'PreflopVCom', 'FlopVCom', 'TurnVCom', 'FlopAcB',
                         'FlopAcC',
                         'FlopAcK', 'FlopAcR', 'FlopVSim', 'PreflopAcB', 'PreflopAcC', 'PreflopAcK', 'PreflopAcR',
                         'RiverAcB',
                         'RiverAcC', 'RiverAcK', 'RiverAcR', 'RiverVSim', 'TurnAcB', 'TurnAcC', 'TurnAcK', 'TurnAcR',
                         'TurnVSim']
    X = df[all_predictors]

    # Reg
    # vif
    # intercept = [1] * df['PreflopBet'].__len__()
    # design_matrix = pd.np.column_stack([df['PreflopBet'], df['FlopBet'], df['TurnBet'], df['RiverBet'], df['PreflopVCom'],
    #                                     df['FlopVCom'], df['TurnVCom'], df['FlopAcB'], df['FlopAcC'], df['FlopAcK'],
    #                                     df['FlopAcR'], df['FlopVSim'], df['PreflopAcB'], df['PreflopAcC'],
    #                                     df['PreflopAcK'], df['PreflopAcR'], df['RiverAcB'], df['RiverAcC'],
    #                                     df['RiverAcK'], df['RiverAcR'], df['RiverVSim'], df['TurnAcB'],
    #                                     df['TurnAcC'], df['TurnAcK'], df['TurnAcR'],df['TurnVSim'], intercept])
    # RiverBet 367.5
    # TurnBet 98.4
    # FlopBet 33.8
    # design_matrix = pd.np.column_stack(
    #     [df['PreflopBet'], df['PreflopVCom'],
    #      df['FlopVCom'], df['TurnVCom'], df['FlopAcB'], df['FlopAcC'], df['FlopAcK'],
    #      df['FlopAcR'], df['FlopVSim'], df['PreflopAcB'], df['PreflopAcC'],
    #      df['PreflopAcK'], df['PreflopAcR'], df['RiverAcB'], df['RiverAcC'],
    #      df['RiverAcK'], df['RiverAcR'], df['RiverVSim'], df['TurnAcB'],
    #      df['TurnAcC'], df['TurnAcK'], df['TurnAcR'], df['TurnVSim'], intercept])
    # vif = [variance_inflation_factor(design_matrix, i) for i in range(design_matrix.shape[1])]
    # index, value = max(enumerate(vif), key=operator.itemgetter(1))
    # print(vif)
    # print(index, value)

    # un_col_predictors = (vif_analysis(df, all_predictors, 5))
    # print(un_col_predictors)

    pca_reduced_X = feature_size_reduce(X)

    # y = df[['RiverVCom']]
    # X_train = X.ix[:9999, :].values
    # # X_train = pca_reduced_X[:10000]
    # y_train = y.ix[:9999, ].values.ravel()
    # X_test = X.ix[10000:19999, :].values
    # # X_test = pca_reduced_X[10000:20000]
    # y_test = y.ix[10000:19999, ].values.ravel()
    # optunity_tune_svr_rbf(X_train, y_train, X_test, y_test)


def feature_size_reduce(X):
    pca = decomposition.PCA()
    pca.fit(X)
    plt.semilogy(pca.explained_variance_ratio_, '--o')
    plt.show()
    print(pca.explained_variance_)
    pca.n_components = 14
    X_reduced = pca.fit_transform(X)
    return X_reduced


def analyze_player():
    df = pd.read_csv('players.csv')
    # df1 = df.assign(f=df['BankIncBet']/df['TotalGames']).sort_values(by='f', ascending=False).drop('f', axis=1)
    # print df1.head(10)
    # df1 = df.assign(f=df['TotalR']/df['TotalGames']).sort_values(by='f', ascending=False).drop('f', axis=1)
    # print df1.head(10)
    df1 = df.loc[df['TotalGames'] >= 5]
    df1['AvgIncBet'] = df1['BankIncBet'] / df1['TotalGames']

    avg = df1.loc[df['Player'] == 'AVERAGE']
    threshold = float(avg.get('AvgIncBet'))

    df1['AvgBet'] = df1['TotalBet'] / df1['TotalGames']
    df1['AvgB'] = df1['TotalB'] / df1['TotalGames']
    df1['AvgR'] = df1['TotalR'] / df1['TotalGames']
    df1['AvgC'] = df1['TotalC'] / df1['TotalGames']
    df1['AvgK'] = df1['TotalK'] / df1['TotalGames']

    # SVM
    # X = df1[['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'PreflopVCom', 'FlopVCom', 'TurnVCom',
    #          'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']]

    # X = df1[['AvgR', 'AvgC', 'RiverVCom', 'FlopVSim']]
    # y = df1.apply(lambda x: 1 if x.get('AvgIncBet') > threshold else 0, axis=1)
    # dump_svmlight_file(X, y, 'svmlight_player.dat', zero_based=True, multilabel=False)
    # y, x = svm_read_problem('svmlight_player.dat')
    # m = svm_train(y[0:800], x[0:800], '-c 1 -h 0 -t 0')
    # p_label, p_acc, p_val = svm_predict(y[800:], x[800:], m)


    # # Reg
    # # vif
    # # AvgBet, FlopVCom, TurnVCom
    # intercept = [1] * df1['AvgIncBet'].__len__()
    # design_matrix = pd.np.column_stack([df1['AvgB'], df1['AvgR'], df1['AvgC'], df1['AvgK'],
    #                                     df1['PreflopVCom'], df1['RiverVCom'],
    #                                     df1['FlopVSim'], df1['TurnVSim'], df1['RiverVSim'], intercept])
    #
    # vif = [variance_inflation_factor(design_matrix, i) for i in range(design_matrix.shape[1])]
    # print vif

    # # boxcox
    # df1['AvgIncBet'] = df1['AvgIncBet'] + 0.00001
    # df1['TransAvgIncBet'], bet_lambda = stats.boxcox(df1['AvgIncBet'])
    # print bet_lambda

    # # model
    # bet_size_model = ols('TransAvgIncBet ~ AvgBet + AvgB + AvgR + AvgC + AvgK + PreflopVCom + FlopVCom + TurnVCom + '
    #                      'RiverVCom + FlopVSim + TurnVSim + RiverVSim', data=df1).fit()
    # print bet_size_model.summary()
    # bet_size_model = ols('AvgIncBet ~ AvgB + AvgR + AvgC + AvgK + PreflopVCom + '
    #                      'RiverVCom + FlopVSim + TurnVSim + RiverVSim', data=df1).fit()
    # print bet_size_model.summary()

    # bet_size_model = ols('AvgIncBet ~  AvgR + AvgC ', data=df1).fit()
    # print bet_size_model.summary()

    # AvgR, AvgC, RiverVCom, FlopVSim


def analyze_game():
    # df = pd.read_csv('data.csv')

    # SVM
    # X = df1[['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'PreflopVCom', 'FlopVCom', 'TurnVCom',
    #          'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']]

    # X = df[['PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']]
    # y = df['IsWin']
    # dump_svmlight_file(X, y, 'svmlight_game.dat', zero_based=True, multilabel=False)

    # y, x = svm_read_problem('svmlight_game.dat')
    # m = svm_train(y[0:5000], x[0:5000], '-c 1 -h 0 -t 0')
    # p_label, p_acc, p_val = svm_predict(y[5000:10000], x[5000:10000], m)

    # print m.get_sv_coef()
    # print m.get_SV()
    #

    # model
    lef = ['PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom, FlopVSim']
    # bet_size_model = ols('IsWin ~ PreflopVCom + FlopVCom + TurnVCom + RiverVCom + FlopVSim + TurnVSim + RiverVSim',
    #                      data=df).fit()
    # print bet_size_model.summary()


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    # _clean()
    # _player()
    # analyze_player()
    # analyze_game()
    # tune_player()
    tune_river_value_com()
    # df = pd.read_csv("actions.csv")
    # df1 = df[df['Action'].str.contains('RiverVCom')]
    # for index, row in df1.iterrows():
    #     print(row['Action'].replace('RiverVCom','').replace(',','').replace(' ',''))