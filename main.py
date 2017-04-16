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
        model = sklearn.svm.SVC(kernel='rbf', C=para[0], gamma=10 ** para[1])
        model.fit(x_train, y_train)
        decision_values = model.decision_function(x_test)
        auc = optunity.metrics.roc_auc(y_test, decision_values)
        return auc

    @optunity.cross_validated(x=X_train_all, y=y_train_all, num_folds=5)
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, C, loggamma):
        acc = train1(x_train, y_train, x_test, y_test, (C, loggamma))
        # print("train acc: {}".format(acc))
        return acc

    optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=50, C=[0, 10], loggamma=[-5, 0],
                                                  pmap=optunity.pmap)
    df = optunity.call_log2dataframe(info.call_log)
    df = df.sort_values(by='value', ascending=False)[:10]
    df1 = df.apply(lambda x: pd.Series({'True Acc':
                                            train1(X_train_all, y_train_all, x_test, y_test,
                                                   (x.loc['C'], x.loc['loggamma']))
                                        }), axis=1)
    df3 = pd.concat([df, df1], axis=1)
    return train1(X_train_all, y_train_all, x_test, y_test, (optimal_rbf_pars['C'], optimal_rbf_pars['loggamma']))


def optunity_tune_svr_rbf(X_train_all, y_train_all):
    outer_cv = optunity.cross_validated(x=X_train_all, y=y_train_all, num_folds=3)

    def compute_mse_rbf_tuned(x_train, y_train, x_test, y_test):
        """Computes MSE of an SVR with RBF kernel and optimized hyperparameters."""

        # define objective function for tuning
        @optunity.cross_validated(x=x_train, y=y_train, num_iter=2, num_folds=5)
        def tune_cv(x_train, y_train, x_test, y_test, C, gamma):
            model = sklearn.svm.SVR(C=C, gamma=gamma)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            return optunity.metrics.mse(y_test, predictions)

        # optimize parameters
        optimal_pars, _, _ = optunity.minimize(tune_cv, 150, C=[1, 100], gamma=[0, 50], pmap=optunity.pmap)
        print("optimal hyperparameters: " + str(optimal_pars))
        tuned_model = sklearn.svm.SVR(**optimal_pars)
        tuned_model.fit(x_train, y_train)
        predictions = tuned_model.predict(x_test)
        return optunity.metrics.mse(y_test, predictions)

    # wrap with outer cross-validation
    compute_mse_rbf_tuned = outer_cv(compute_mse_rbf_tuned)
    return  compute_mse_rbf_tuned()



def tune_player_svr(df, top, para=2):
    res = []
    for predictors in itertools.combinations(
            ['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'FlopVSim', 'TurnVSim', 'RiverVSim', 'FlopVCom',
             'TurnVCom', 'Aggressive'], para):
        X = df[list(predictors)]
        y = df.apply(lambda x: 1 if x.get('Player') in top else 0, axis=1)
        X_train = X.ix[:, :].values
        y_train = y.ix[:, ].values
        result = (predictors, optunity_tune_svr_rbf(X_train, y_train))
        print(result)
        res.append(result)
    return sorted(res, key=lambda x: x[1], reverse=True)



def tune_player_svc(df, top, para=2):
    res = []
    for predictors in itertools.combinations(
            ['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'FlopVSim', 'TurnVSim', 'RiverVSim', 'FlopVCom',
             'TurnVCom', 'RiverVCom', 'Aggressive'], para):
        X = df[list(predictors)]
        y = df.apply(lambda x: 1 if x.get('Player') in top else 0, axis=1)
        X_train = X.ix[:len(df.index) / 2 - 1, :].values
        y_train = y.ix[:len(df.index) / 2 - 1, ].values
        X_test = X.ix[len(df.index) / 2:, :].values
        y_test = y.ix[len(df.index) / 2:, ].values
        result = (predictors, optunity_tune_svm_rbf(X_train, y_train, X_test, y_test))
        print(result)
        res.append(result)
    return sorted(res, key=lambda x: x[1], reverse=True)


def tune_player():
    df = pd.read_csv('players.csv')
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalGames']
    df['AvgBet'] = df['TotalBet'] / df['TotalGames']
    df['AvgB'] = df['TotalB'] / df['TotalGames']
    df['AvgR'] = df['TotalR'] / df['TotalGames']
    df['AvgC'] = df['TotalC'] / df['TotalGames']
    df['AvgK'] = df['TotalK'] / df['TotalGames']
    df['Aggressive'] = (df['TotalB'] + df['TotalR']) / (df['TotalC'] + df['TotalK'])

    top33, top50, low33, low50 = player_classify(type=1)

    df50 = df[df['Player'].isin(top50 + low50)]
    df33 = df[df['Player'].isin(top33 + low33)]

    result50_para1 = tune_player_svc(df50, top50, para=1)
    plt.plot([x[1] for x in list(reversed(result50_para1))], 'b^', label='parameter1 top50%')

    result33_para1 = tune_player_svc(df33, top33, para=1)
    plt.plot([x[1] for x in list(reversed(result33_para1))], 'g^', label='parameter1 top33%')

    result50_para2 = tune_player_svc(df50, top50, para=2)
    plt.plot([x[1] for x in list(reversed(result50_para2))], 'bo', label='parameter2 top50%')

    result33_para2 = tune_player_svc(df33, top33, para=2)
    plt.plot([x[1] for x in list(reversed(result33_para2))], 'go', label='parameter2 top33%')

    result50_para3 = tune_player_svc(df50, top50, para=3)
    plt.plot([x[1] for x in list(reversed(result50_para3))], 'b-', label='parameter3 top50%')

    result33_para3 = tune_player_svc(df33, top33, para=3)
    plt.plot([x[1] for x in list(reversed(result33_para3))], 'g-', label='parameter3 top33%')

    plt.legend(loc=0)
    plt.title('Player Performance Classify Accuracy')
    plt.savefig('Player Performance Classify Accuracy2.png')
    plt.show()

    print('top50%Winning Results: {}'.format(result50_para1))
    print('top50%Winning Results: {}'.format(result50_para2))
    print('top50%Winning Results: {}'.format(result50_para3))

    print('top33%Winning Results: {}'.format(result33_para1))
    print('top33%Winning Results: {}'.format(result33_para2))
    print('top33%Winning Results: {}'.format(result33_para3))


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
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalGames']

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


def handvalue_boxplot():
    from pylab import plot, show, savefig, xlim, figure, \
        hold, ylim, legend, boxplot, setp, axes

    # function for setting the colors of the box plots pairs
    def setBoxColors(bp):
        setp(bp['boxes'][0], color='blue')
        setp(bp['caps'][0], color='blue')
        setp(bp['caps'][1], color='blue')
        setp(bp['whiskers'][0], color='blue')
        setp(bp['whiskers'][1], color='blue')
        setp(bp['fliers'][0], markeredgecolor='blue')
        setp(bp['medians'][0], color='blue')

        setp(bp['boxes'][1], color='red')
        setp(bp['caps'][2], color='red')
        setp(bp['caps'][3], color='red')
        setp(bp['whiskers'][2], color='red')
        setp(bp['whiskers'][3], color='red')
        setp(bp['fliers'][1], markeredgecolor='red')
        setp(bp['medians'][1], color='red')

    df = pd.read_csv("data.csv")
    A = [df['FlopVSim'], df['FlopVCom']]
    B = [df['TurnVSim'], df['TurnVCom']]
    C = [df['RiverVSim'], df['RiverVCom']]

    fig = figure()
    ax = axes()

    # first boxplot pair
    bp = boxplot(A, positions=[1, 2], widths=0.6)
    setBoxColors(bp)

    # second boxplot pair
    bp = boxplot(B, positions=[4, 5], widths=0.6)
    setBoxColors(bp)

    # thrid boxplot pair
    # Note, useed whis = range to let boxplot include all the data
    bp = boxplot(C, positions=[7, 8], whis="range", widths=0.6)
    setBoxColors(bp)

    # set axes limits and labels
    xlim(0, 9)
    ylim(0, 1)
    ax.set_xticklabels(['Flop', 'Turn', 'River'])
    ax.set_xticks([1.5, 4.5, 7.5])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    legend((hB, hR), ('Simple', 'Complex'))
    hB.set_visible(False)
    hR.set_visible(False)
    fig.canvas.set_window_title('handvalue_stage_plot')
    plt.title('handvalue_stage_plot')
    savefig('handvalue_stage_plot.png')
    show()


def player_classify(type=0):
    df = pd.read_csv("players.csv")
    df1 = df.loc[df['TotalGames'] >= 50]
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalGames']
    df1.sort_values(by=['AvgIncBet'], axis=0, inplace=True, ascending=False)
    size = df1['Player'].count()
    if type==0:
        top50 = list(df1['Player'].head(int(size / 2)))
        top25 = list(df1['Player'].head(int(size / 4)))
        top10 = list(df1['Player'].head(int(size / 10)))

        low50 = list(df1['Player'].tail(int(size / 2)))
        low25 = list(df1['Player'].tail(int(size / 4)))
        low10 = list(df1['Player'].tail(int(size / 10)))
        return top10, top25, top50, low10, low25, low50
    else:
        top50 = list(df1['Player'].head(int(size / 2)))
        top33 = list(df1['Player'].head(int(size / 3)))

        low50 = list(df1['Player'].tail(int(size / 2)))
        low33 = list(df1['Player'].tail(int(size / 3)))
        return top33, top50, low33, low50


def handvalue_boxplot_stage(stage_name):
    from pylab import plot, show, savefig, xlim, figure, \
        hold, ylim, legend, boxplot, setp, axes

    # function for setting the colors of the box plots pairs
    def setBoxColors(bp):
        setp(bp['boxes'][0], color='blue')
        setp(bp['caps'][0], color='blue')
        setp(bp['caps'][1], color='blue')
        setp(bp['whiskers'][0], color='blue')
        setp(bp['whiskers'][1], color='blue')
        setp(bp['fliers'][0], markeredgecolor='blue')
        setp(bp['medians'][0], color='blue')

        setp(bp['boxes'][1], color='red')
        setp(bp['caps'][2], color='red')
        setp(bp['caps'][3], color='red')
        setp(bp['whiskers'][2], color='red')
        setp(bp['whiskers'][3], color='red')
        setp(bp['fliers'][1], markeredgecolor='red')
        setp(bp['medians'][1], color='red')

    df = pd.read_csv("data.csv")
    top10, top25, top50, low10, low25, low50 = player_classify()

    A = [df[df['Player'].isin(top50)]['{}VCom'.format(stage_name)],
         df[df['Player'].isin(low50)]['{}VCom'.format(stage_name)]]
    B = [df[df['Player'].isin(top25)]['{}VCom'.format(stage_name)],
         df[df['Player'].isin(low25)]['{}VCom'.format(stage_name)]]
    C = [df[df['Player'].isin(top10)]['{}VCom'.format(stage_name)],
         df[df['Player'].isin(low10)]['{}VCom'.format(stage_name)]]

    fig = figure()
    ax = axes()

    # first boxplot pair
    bp = boxplot(A, positions=[1, 2], widths=0.6)
    setBoxColors(bp)

    # second boxplot pair
    bp = boxplot(B, positions=[4, 5], widths=0.6)
    setBoxColors(bp)

    # thrid boxplot pair
    # Note, useed whis = range to let boxplot include all the data
    bp = boxplot(C, positions=[7, 8], widths=0.6)
    setBoxColors(bp)

    # set axes limits and labels
    xlim(0, 9)
    ylim(0, 1)
    ax.set_xticklabels(['50%', '25%', '10%'])
    ax.set_xticks([1.5, 4.5, 7.5])

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    legend((hB, hR), ('Top', 'Low'))
    hB.set_visible(False)
    hR.set_visible(False)
    fig.canvas.set_window_title('{}_compare_plot'.format(stage_name))
    plt.title('{}_compare_plot'.format(stage_name))
    savefig('handvalue_{}_player_compare_plot.png'.format(stage_name))
    show()


def profit_plot():
    df = pd.read_csv('players.csv')
    df1 = df.loc[df['TotalGames'] >= 50]
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalBet']

    print(sum(df1['AvgIncBet']) / len(df1.index))
    print(numpy.std(df1['AvgIncBet']))

    df1['AvgBet'] = df1['TotalBet'] / df1['TotalGames']

    plt.plot(df1['AvgIncBet'], df1['AvgBet'], 'ro')
    plt.ylabel('AvgBet')
    plt.xlabel('AvgIncBet')
    plt.title('AvgBet-AvgIncBet PerGame')
    plt.show()
    plt.savefig('AvgBet-AvgIncBet PerGame.png')

    plt.hist(df1['AvgIncBet'], bins=20)
    plt.title('AvgIncBet PerGame')
    plt.ylabel('AvgIncBet')
    plt.show()
    plt.savefig('AvgIncBet PerGame')


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    # _clean()
    # _player()
    # analyze_player()
    # analyze_game()

    tune_player()
    # handvalue_boxplot()
    # handvalue_boxplot_stage('Flop')
    # handvalue_boxplot_stage('Turn')
    # handvalue_boxplot_stage('River')
    # profit_plot()
