from collections import defaultdict
import numpy
import sklearn.svm
from sklearn.pipeline import Pipeline

import get_percentage as gp
import pandas as pd
from statsmodels.formula.api import ols
from scipy import stats
import itertools
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import dump_svmlight_file
import optunity
import optunity.metrics
from sklearn import decomposition, linear_model
import operator
import tkinter
import matplotlib.pyplot as plt
import sklearn.naive_bayes as nb


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


def optunity_tune_logit(X_train_all, y_train_all, x_test, y_test):
    def train1(x_train, y_train, x_test, y_test, para):
        model = linear_model.LogisticRegression(penalty='l1', C=10**para)
        model.fit(x_train, y_train)
        p_label = model.predict(x_test)
        acc_arr = [1 for i in range(len(p_label)) if p_label[i] == y_test[i]]
        acc = float(sum(acc_arr)) / len(p_label)
        return acc

    @optunity.cross_validated(x=X_train_all, y=y_train_all, num_folds=5)
    def svm_rbf_tuned_auroc(x_train, y_train, x_test, y_test, logC):
        acc = train1(x_train, y_train, x_test, y_test, logC)
        # print("train acc: {}".format(acc))
        return acc

    optimal_rbf_pars, info, _ = optunity.maximize(svm_rbf_tuned_auroc, num_evals=50, logC=[0, 3],
                                                  pmap=optunity.pmap)
    df = optunity.call_log2dataframe(info.call_log)
    df = df.sort_values(by='value', ascending=False)[:10]
    df1 = df.apply(lambda x: pd.Series({'True Acc':
                                            train1(X_train_all, y_train_all, x_test, y_test,
                                                   x['logC'])
                                        }), axis=1)
    df3 = pd.concat([df, df1], axis=1)
    return df3['value'].iloc[0], df3['True Acc'].iloc[0]


def tune_player_logit():
    df = pd.read_csv('players.csv')
    df = df.loc[df['TotalGames'] >= 50]
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalGames']
    df['AvgBet'] = df['TotalBet'] / df['TotalGames']
    df['AvgB'] = df['TotalB'] / df['TotalGames']
    df['AvgR'] = df['TotalR'] / df['TotalGames']
    df['AvgC'] = df['TotalC'] / df['TotalGames']
    df['AvgK'] = df['TotalK'] / df['TotalGames']
    df['Aggressive'] = (df['TotalC'] + df['TotalB'] + df['TotalR']) / (df['TotalK'])

    top10, top25, top50, low10, low25, low50 = player_classify()
    df10 = df[df['Player'].isin(top10 + low10)]
    df10 = df10.reset_index(drop = True)
    all_predictors = ['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'FlopVSim', 'TurnVSim', 'RiverVSim', 'FlopVCom', 'RiverVCom',
             'TurnVCom', 'Aggressive']
    res = []
    for L in range(all_predictors.__len__(), all_predictors.__len__() + 1):
        for predictors in itertools.combinations(all_predictors, L):
            X = df10[list(predictors)]
            y = df10.apply(lambda x: 1 if x.get('Player') in top10 else 0, axis=1)
            separ = int(len(df10.index) * 7 / 10)
            X_train = X.ix[:separ - 1, :].values
            y_train = y.ix[:separ - 1, ].values
            X_test = X.ix[separ:, :].values
            y_test = y.ix[separ:, ].values
            p = optunity_tune_svc_rbf(X_train, y_train, X_test, y_test)
            result = (predictors, p[0], p[1])
            print(result)
            res.append(result)
    print(sorted(res, key=lambda x: x[0], reverse=True))


def naive_bayes_clas(X_all, y_all):
    def train1(x_train, y_train, x_test, y_test, alpha):
        nbm = nb.MultinomialNB(alpha=alpha)
        model = nbm.fit(x_train, y_train)
        p_label = model.predict(x_test)
        acc_arr = [1 for i in range(len(p_label)) if p_label[i] == y_test[i]]
        acc = float(sum(acc_arr)) / len(p_label)
        return acc

    @optunity.cross_validated(x=X_all, y=y_all, num_folds=10)
    def naive_tune(x_train, y_train, x_test, y_test, alpha):
        acc = train1(x_train, y_train, x_test, y_test, alpha)
        return acc

    optimal_pars, info, _ = optunity.maximize(naive_tune, num_evals=50, alpha=[0, 10],
                                              pmap=optunity.pmap)
    df = optunity.call_log2dataframe(info.call_log)
    df = df.sort_values(by='value', ascending=False)[:10]
    return list(df['value'])[0]


def tune_player_nb():
    df = pd.read_csv('players.csv')
    df = df.loc[df['TotalGames'] >= 50]
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalGames']
    df['AvgBet'] = df['TotalBet'] / df['TotalGames']
    df['AvgB'] = df['TotalB'] / df['TotalGames']
    df['AvgR'] = df['TotalR'] / df['TotalGames']
    df['AvgC'] = df['TotalC'] / df['TotalGames']
    df['AvgK'] = df['TotalK'] / df['TotalGames']
    df['Aggressive'] = (df['TotalC'] + df['TotalB'] + df['TotalR']) / (df['TotalK'])

    top10, top25, top50, low10, low25, low50 = player_classify()
    df10 = df[df['Player'].isin(top10 + low10)]
    df10 = df10.reset_index(drop=True)
    all_predictors = ['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'Aggressive']
    res = []
    for L in range(1, all_predictors.__len__() + 1):
        for predictors in itertools.combinations(all_predictors, L):
            X = df10[list(predictors)]
            y = df10.apply(lambda x: 1 if x.get('Player') in top10 else -1, axis=1)
            X_all = X.values
            y_all = y.values
            result = naive_bayes_clas(X_all, y_all)
            res.append([result, list(predictors)])
            print(result)
    print(sorted(res, key=lambda x: x[0], reverse=True))


def optunity_tune_svc_rbf(X_train_all, y_train_all, x_test, y_test):
    def train1(x_train, y_train, x_test, y_test, para):
        model = sklearn.svm.SVC(kernel='rbf', C=para[0], gamma=10 ** para[1])
        model.fit(x_train, y_train)
        p_label = model.predict(x_test)
        acc_arr = [1 for i in range(len(p_label)) if p_label[i] == y_test[i]]
        acc = float(sum(acc_arr)) / len(p_label)
        return acc

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
                                                   (x['C'], x['loggamma']))
                                        }), axis=1)
    df3 = pd.concat([df, df1], axis=1)
    return df3['value'].iloc[0], df3['True Acc'].iloc[0], df3['C'].iloc[0], df3['loggamma'].iloc[0]


def tune_player_svc(df, top, para=2, clump=False):
    res = []
    for predictors in itertools.combinations(
            ['AvgBet', 'AvgB', 'AvgR', 'AvgC', 'AvgK', 'FlopVSim', 'TurnVSim', 'RiverVSim', 'FlopVCom', 'RiverVCom',
             'TurnVCom', 'Aggressive'], para):
        X = df[list(predictors)]
        y = df.apply(lambda x: 1 if x.get('Player') in top else 0, axis=1)
        separ = int(len(df.index) * 7 / 10)
        X_train = X.ix[:separ - 1, :].values
        y_train = y.ix[:separ - 1, ].values
        X_test = X.ix[separ:, :].values
        y_test = y.ix[separ:, ].values
        p = optunity_tune_svc_rbf(X_train, y_train, X_test, y_test)
        result = (predictors, p[0], p[1], p[2], p[3])
        print(result)
        res.append(result)
    if clump:
        return sorted(res, key=lambda x: x[1], reverse=True)[:5]
    return sorted(res, key=lambda x: x[1], reverse=True)


def tune_player_all():
    df = pd.read_csv('players.csv')
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalGames']
    df['AvgBet'] = df['TotalBet'] / df['TotalGames']
    df['AvgB'] = df['TotalB'] / df['TotalGames']
    df['AvgR'] = df['TotalR'] / df['TotalGames']
    df['AvgC'] = df['TotalC'] / df['TotalGames']
    df['AvgK'] = df['TotalK'] / df['TotalGames']
    df['Aggressive'] = (df['TotalC'] + df['TotalB'] + df['TotalR']) / (df['TotalK'])

    top10, _, _, low10, _, _ = player_classify()

    df10 = df[df['Player'].isin(top10 + low10)]
    df10 = df10.reset_index(drop=True)
    res = []
    for i in range(1, 13):
        res.append(tune_player_svc(df10, top10, para=i, clump=True))
    for r in res:
        print('top10%Winning Results: {}'.format(r))


def tune_player():
    df = pd.read_csv('players.csv')
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalGames']
    df['AvgBet'] = df['TotalBet'] / df['TotalGames']
    df['AvgB'] = df['TotalB'] / df['TotalGames']
    df['AvgR'] = df['TotalR'] / df['TotalGames']
    df['AvgC'] = df['TotalC'] / df['TotalGames']
    df['AvgK'] = df['TotalK'] / df['TotalGames']
    df['Aggressive'] = (df['TotalC'] + df['TotalB'] + df['TotalR']) / (df['TotalK'])

    top10, top25, top50, low10, low25, low50 = player_classify()

    df50 = df[df['Player'].isin(top50 + low50)]
    df25 = df[df['Player'].isin(top25 + low25)]
    df10 = df[df['Player'].isin(top10 + low10)]
    df10 = df10.reset_index(drop=True)
    df25 = df25.reset_index(drop=True)
    df50 = df50.reset_index(drop=True)
    result10_para1 = tune_player_svc(df10, top10, para=1, clump=True)
    result10_para2 = tune_player_svc(df10, top10, para=2, clump=True)
    result10_para3 = tune_player_svc(df10, top10, para=3, clump=True)
    result10_para4 = tune_player_svc(df10, top10, para=4, clump=True)
    print('top10%Winning Results: {}'.format(result10_para1))
    print('top10%Winning Results: {}'.format(result10_para2))
    print('top10%Winning Results: {}'.format(result10_para3))
    print('top10%Winning Results: {}'.format(result10_para4))

    # result50_para1 = tune_player_svc(df50, top50, para=1)
    # # plt.plot([x[1] for x in list(reversed(result50_para1))], 'b^', label='parameter1 top50%')
    #
    # result25_para1 = tune_player_svc(df25, top25, para=1)
    # # plt.plot([x[1] for x in list(reversed(result33_para1))], 'g^', label='parameter1 top33%')
    #
    # result50_para2 = tune_player_svc(df50, top50, para=2)
    # # plt.plot([x[1] for x in list(reversed(result50_para2))], 'bo', label='parameter2 top50%')
    #
    # result25_para2 = tune_player_svc(df25, top25, para=2)
    # # plt.plot([x[1] for x in list(reversed(result33_para2))], 'go', label='parameter2 top33%')
    #
    # result50_para3 = tune_player_svc(df50, top50, para=3)
    # # plt.plot([x[1] for x in list(reversed(result50_para3))], 'b-', label='parameter3 top50%')
    #
    # result25_para3 = tune_player_svc(df25, top25, para=3)
    # # plt.plot([x[1] for x in list(reversed(result33_para3))], 'g-', label='parameter3 top33%')
    # #
    # # plt.legend(loc=0)
    # # plt.title('Player Performance Classify Accuracy')
    # # plt.savefig('Player Performance Classify Accuracy2.png')
    # # plt.show()
    # #
    # print('top50%Winning Results: {}'.format(result50_para1))
    # print('top50%Winning Results: {}'.format(result50_para2))
    # print('top50%Winning Results: {}'.format(result50_para3))
    #
    # print('top25%Winning Results: {}'.format(result25_para1))
    # print('top25%Winning Results: {}'.format(result25_para2))
    # print('top25%Winning Results: {}'.format(result25_para3))


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
    ax.set_ylabel('Hand Value')

    # draw temporary red and blue lines and use them to create a legend
    hB, = plot([1, 1], 'b-')
    hR, = plot([1, 1], 'r-')
    legend((hB, hR), ('Basic', 'Effective'))
    hB.set_visible(False)
    hR.set_visible(False)
    fig.canvas.set_window_title('handvalue_stage_plot')
    plt.title('Two types of hand value at different stage')
    savefig('handvalue_stage_plot.png')
    show()


def player_classify(type=0):
    df = pd.read_csv("players.csv")
    df1 = df.loc[df['TotalGames'] >= 50]
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalGames']
    df1.sort_values(by=['AvgIncBet'], axis=0, inplace=True, ascending=False)
    size = df1['Player'].count()
    if type == 0:
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

    print("average")
    print(sum(df1['AvgIncBet']) / len(df1.index))

    print("sd")
    print(numpy.std(df1['AvgIncBet']))

    plt.hist(df1['AvgIncBet'], bins=20)
    plt.title('Winning Ability of Players')
    plt.xlabel('Profit')
    plt.show()

    print("Win most")
    print(df1.head())
    print("Lose Most")
    print(df1.tail())


def aggressive_analysis():
    df = pd.read_csv('players.csv')
    df1 = df.loc[df['TotalGames'] >= 50]
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalBet']

    df1['Aggressive'] = (df1['TotalC'] + df1['TotalB'] + df1['TotalR']) / (df1['TotalK'])
    df1 = df1.sort_values(by='Aggressive', ascending=False)
    top10, top25, top50, low10, low25, low50 = player_classify()

    df_top = df1[df1['Player'].isin(top10)]
    df_low = df1[df1['Player'].isin(low10)]
    df_top = df_top.sort_values(by='AvgIncBet', ascending=False)
    df_low = df_low.sort_values(by='AvgIncBet', ascending=True)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title('Aggressiveness')
    num = int(df_top['Player'].count())
    ax.set_xticklabels(['100%', '0%'])
    ax.set_xticks([0, num])
    plt.xlim(0, num)

    _top = list(df_top['Aggressive'].values)
    _low = list(df_low['Aggressive'].values)
    plt.plot(_top, 'ro', label='Winning Players')
    plt.plot(_low, 'bo', label='Losing Players')
    plt.legend(loc=0)
    plt.ylabel("Aggressiveness")

    plt.xlabel("Percentile")

    plt.show()

    print(stats.ttest_ind(_top, _low))

    print("Most Aggr")
    print(df1.head())
    print("Least Aggr")
    print(df1.tail())


def X_vs_profit(X):
    df = pd.read_csv('players.csv')
    df1 = df.loc[df['TotalGames'] >= 50]
    df1['AvgIncBet'] = (df1['BankIncBet'] - df1['TotalBet']) / df1['TotalBet']
    df1['Aggressive'] = (df1['TotalC'] + df1['TotalB'] + df1['TotalR']) / (df1['TotalK'])


    df1 = df1.sort_values(by='AvgIncBet', ascending=True)

    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_title('{}-Profit'.format(X))

    model = ols('AvgIncBet ~ {}'.format(X), data=df1).fit()
    print(model.summary())
    plt.plot(df1['AvgIncBet'], df1[X], 'ro')
    plt.ylabel(X)
    plt.xlabel("Profit")
    plt.show()


def totalgame_analysis():
    df = pd.read_csv('players.csv')
    df = df.loc[df['TotalGames'] >= 50]
    df = df.sort_values(by='TotalGames', ascending=False)
    df['Aggressive'] = (df['TotalB'] + df['TotalR']) / (df['TotalC'] + df['TotalK'])
    df['AvgIncBet'] = (df['BankIncBet'] - df['TotalBet']) / df['TotalBet']

    df = df.sort_values(by='TotalGames', ascending=False)
    top10, top25, top50, low10, low25, low50 = player_classify()

    df_top = df[df['Player'].isin(top10)]
    df_low = df[df['Player'].isin(low10)]

    _top = list(df_top['TotalGames'].values)
    _low = list(df_low['TotalGames'].values)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1)

    ax.set_yscale('log')

    plt.plot(_top, 'ro', label='Winning Players')
    plt.plot(_low, 'bo', label='Losing Players')
    plt.title('TotalGames')
    plt.legend(loc=0)
    plt.show()
    plt.savefig('TotalGames.png')

    print(stats.ttest_ind(_top, _low))
    print('Most Games')
    print(df.head())
    print('Least Games')
    print(df.tail())


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    # _clean()
    # _player()

    # handvalue_boxplot()
    # handvalue_boxplot_stage('Flop')
    # handvalue_boxplot_stage('Turn')
    # handvalue_boxplot_stage('River')
    # profit_plot()
    aggressive_analysis()
    # totalgame_analysis()
    # tune_player_multi_svc()


    # X_vs_profit('RiverVCom')
    # X_vs_profit('RiverVSim')
    # X_vs_profit('TurnVCom')
    # X_vs_profit('TurnVSim')
    # X_vs_profit('FlopVCom')
    # X_vs_profit('FlopVSim')

    # tune_player()
    # tune_player_nb()
    # tune_player_logit()
    # tune_player_all()


