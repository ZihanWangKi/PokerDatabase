from collections import defaultdict
from svmutil import *
import numpy
import get_percentage as gp
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import dump_svmlight_file


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
            l = [0 for x in xrange(16)]
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
    avg = [0 for x in xrange(16)]

    for k, v in player_map.iteritems():
        lst.append([k] + v)
        for i_ in xrange(v.__len__()):
            avg[i_] += v[i_]
    for i_ in xrange(avg.__len__()):
        avg[i_] = (avg[i_] + 0.0) / lst.__len__()

    lst.append(['AVERAGE'] + avg)
    labels = ['Player', 'BankIni', 'BankInc', 'BankIncBet', 'TotalGames', 'TotalBet', 'TotalB', 'TotalR', 'TotalC',
              'TotalK', 'PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom', 'FlopVSim', 'TurnVSim', 'RiverVSim']
    df1 = pd.DataFrame.from_records(lst, columns=labels)
    df1.to_csv('players.csv', index=False)

    # Average
    for k in xrange(lst.__len__() - 1):
        for i_ in xrange(1, lst[k].__len__()):
            lst[k][i_] -= avg[i_ - 1]
    df2 = pd.DataFrame.from_records(lst, columns=labels)
    df2.to_csv('players_avg.csv', index=False)


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

    X = df1[['AvgR', 'AvgC', 'RiverVCom', 'FlopVSim']]
    y = df1.apply(lambda x: 1 if x.get('AvgIncBet') > threshold else 0, axis=1)
    dump_svmlight_file(X, y, 'svmlight_player.dat', zero_based=True, multilabel=False)
    y, x = svm_read_problem('svmlight_player.dat')
    m = svm_train(y[0:800], x[0:800], '-c 1 -h 0 -t 1')
    p_label, p_acc, p_val = svm_predict(y[800:], x[800:], m)

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

    y, x = svm_read_problem('svmlight_game.dat')
    m = svm_train(y[0:5000], x[0:5000], '-c 1 -h 0 -t 0')
    p_label, p_acc, p_val = svm_predict(y[5000:10000], x[5000:10000], m)

    # print m.get_sv_coef()
    # print m.get_SV()
    #

    # model
    # lef = ['PreflopVCom', 'FlopVCom', 'TurnVCom', 'RiverVCom, FlopVSim']
    # bet_size_model = ols('IsWin ~ PreflopVCom + FlopVCom + TurnVCom + RiverVCom + FlopVSim + TurnVSim + RiverVSim',
    #                      data=df).fit()
    # print bet_size_model.summary()





if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    # _clean()
    # _player()
    # analyze_player()
    analyze_game()
