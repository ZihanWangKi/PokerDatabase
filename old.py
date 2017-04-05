# TODO clean this
from main import *
def simple():
    df = pd.read_csv('poker_data_simple.csv')

    win_tf = df["Won"].map(positive).tolist()
    win_games = list(filter(lambda x: win_tf[x] == 1, xrange(len(win_tf))))
    preflop_value = df['PreflopV'].tolist()
    flop_value = df['FlopV'].tolist()
    turn_value = df['TurnV'].tolist()
    river_value = df['RiverV'].tolist()
    win_length = sum(win_tf)

    # to_show = 150
    # start = np.random.randint(win_games.__len__() - to_show - 1)
    # # plt.plot(range(start + 1, start + to_show + 1), [preflop_value[i] for i in win_games][:to_show], 'o', label="Preflop")
    # # plt.plot(range(start + 1, start + to_show + 1), [flop_value[i] for i in win_games][:to_show], 'o', label="Flop")
    # # plt.plot(range(start + 1, start + to_show + 1), [turn_value[i] for i in win_games][:to_show], 'o', label="Turn")
    # # plt.plot(range(start + 1, start + to_show + 1), [river_value[i] for i in win_games][:to_show], 'o', label="River")
    #
    # plt.plot([river_value[i]-preflop_value[i] for i in win_games[start + 1:start + to_show + 1]][:to_show],
    #          np.zeros(to_show), 'o', label="River-Preflop")
    #
    # plt.legend(loc=0)
    # plt.show()



    win_preflop = defaultdict(float)
    win_flop = defaultdict(float)
    win_turn = defaultdict(float)
    win_river = defaultdict(float)

    for i in win_games:
        win_preflop[preflop_value[i]] += 1.0
        win_flop[flop_value[i]] += 1.0
        win_turn[turn_value[i]] += 1.0
        win_river[river_value[i]] += 1.0

    all_preflop = defaultdict(float)
    all_flop = defaultdict(float)
    all_turn = defaultdict(float)
    all_river = defaultdict(float)
    for i in xrange(len(win_tf)):
        all_preflop[preflop_value[i]] += 1.0
        all_flop[flop_value[i]] += 1.0
        all_turn[turn_value[i]] += 1.0
        all_river[river_value[i]] += 1.0

    for key, value in win_preflop.iteritems():
        value /= all_preflop[key]
        win_preflop[key] = value

    for key, value in win_flop.iteritems():
        value /= all_flop[key]
        win_flop[key] = value

    for key, value in win_turn.iteritems():
        value /= all_turn[key]
        win_turn[key] = value

    for key, value in win_river.iteritems():
        value /= all_river[key]
        win_river[key] = value

    lists = sorted(win_preflop.items())
    x, y = zip(*lists)
    plt.plot(x, y, 'o', label="Preflop")
    # lists = sorted(win_flop.items())
    # x, y = zip(*lists)
    # plt.plot(x, y, 'o', label="Flop")
    # lists = sorted(win_turn.items())
    # x, y = zip(*lists)
    # plt.plot(x, y,  'o', label="Turn")
    # lists = sorted(win_river.items())
    # x, y = zip(*lists)
    # plt.plot(x, y,  'o', label="River")
    plt.legend(loc=0)
    plt.show()


    # df = pd.read_csv('poker_data.csv')
    # df1 = df[['Timestamp','Player', '# of Players', 'Position', '$ Bet', '$ Won', 'PreflopV', 'FlopV', 'TurnV', 'RiverV']]
    # df1.to_csv('poker_data_simple.csv')

    # df = pd.read_csv('poker_data.csv')
    # df1 = mask(df, '# of Players', 2)
    # df1 = df1.reset_index(drop=True)
    # df1.to_csv('poker_data_2_players.csv')

    # df = pd.read_csv('poker_data_2_players.csv')
    # calculator = gp.GetPercentage()
    #
    # df['FlopRV'] = df.apply(lambda row: calculate_real_value_ftr(calculator, row, 0), axis=1)
    # df['TurnRV'] = df.apply(lambda row: calculate_real_value_ftr(calculator, row, 1), axis=1)
    # df['RiverRV'] = df.apply(lambda row: calculate_real_value_ftr(calculator, row, 2), axis=1)
    #
    # df1 = df[['Timestamp','Player', '# of Players', 'Position', '$ Bet', '$ Won', 'PreflopV', 'FlopV', 'TurnV', 'RiverV','FlopRV', 'TurnRV', 'RiverRV']]
    # df1.to_csv('poker_data_2_players.csv')
    # df = pd.read_csv('poker_data_2_players.csv')
    # df1 = df[df['IsWin'] == 1]
    # df1 = df1.reset_index(drop=True)
    # river_value_model = ols("RiverRV ~ FlopRV + TurnRV + PreflopV + RiverV + FlopV + TurnV", data=df1).fit()
    # print river_value_model.summary()
    # df = pd.read_csv('poker_data_2_players.csv')
    # X = df[['RiverRV', 'FlopRV', 'TurnRV', 'RiverV', 'FlopV', 'TurnV', 'PreflopV', 'Bet']]
    # y = df['IsWin']
    # #
    # dump_svmlight_file(X,y,'svmlight.dat',zero_based=True,multilabel=False)

    # y, x = svm_read_problem('svmlight.dat')
    # m = svm_train(y[0:10000], x[0:10000], '-c 2 -h 0 -t 1')
    # p_label, p_acc, p_val = svm_predict(y[12000:15000], x[12000:15000], m)
    # print p_acc



    # df = pd.read_csv('poker_data_2_players.csv')
    # d = range(1500, 2400)
    #
    # plt.plot(d, [df['FlopRV'][a] - df['FlopV'][a] for a in d], label='Flop Difference')
    # # plt.plot(d, df['RiverRV'][500:600], label='River2')
    # plt.legend(loc=0)
    # plt.show()

    # df = pd.read_csv('poker_data_2_players.csv')
    # df['PreflopV'] = df['PreflopV'].apply(lambda x: numpy.log(x))
    # df['FlopV'] = df['FlopV'].apply(lambda x: numpy.log(x))
    # # df['TurnV'] = df['TurnV'].apply(lambda x: numpy.log(x))
    # # df['RiverV'] = df['RiverV'].apply(lambda x: numpy.log(x))
    # df['FlopRV'] = df['FlopRV'].apply(lambda x: numpy.log(x+0.00001))
    # df['TurnRV'] = df['TurnRV'].apply(lambda x: numpy.log(x))
    # df['RiverRV'] = df['RiverRV'].apply(lambda x: numpy.log(x))
    #
    #
    # df['PreflopV'] = (df['PreflopV'] - df['PreflopV'].mean()) / (df['PreflopV'].max() - df['PreflopV'].min())
    # df['FlopV'] = (df['FlopV'] - df['FlopV'].mean()) / (df['FlopV'].max() - df['FlopV'].min())
    # df['TurnV'] = (df['TurnV'] - df['TurnV'].mean()) / (df['TurnV'].max() - df['TurnV'].min())
    # df['RiverV'] = (df['RiverV'] - df['RiverV'].mean()) / (df['RiverV'].max() - df['RiverV'].min())
    # df['FlopRV'] = (df['FlopRV'] - df['FlopRV'].mean()) / (df['FlopRV'].max() - df['FlopRV'].min())
    # df['TurnRV'] = (df['TurnRV'] - df['TurnRV'].mean()) / (df['TurnRV'].max() - df['TurnRV'].min())
    # df['RiverRV'] = (df['RiverRV'] - df['RiverRV'].mean()) / (df['RiverRV'].max() - df['RiverRV'].min())
    #
    #
    # df['Bet'], bet_lambda = stats.boxcox(df['Bet'])

    # intercept = [1]*df['PreflopV'].__len__()
    # design_matrix = pd.np.column_stack([df['PreflopV'], df['FlopV'], df['TurnV'], df['RiverV'], df['FlopRV'],
    #                                     df['TurnRV'], df['RiverRV']])
    # design_matrix = pd.np.column_stack([df['PreflopV'], df['FlopV'], df['RiverV'], df['FlopRV'],
    #                                     df['TurnRV'], df['RiverRV']])
    # vif = [variance_inflation_factor(design_matrix, i) for i in range(design_matrix.shape[1])]
    # print vif

    # turn_size_model = ols("TurnV ~ PreflopV + FlopV + RiverV + FlopRV + TurnRV + RiverRV", data=df).fit()
    # print turn_size_model.summary()

    # bet_size_model = ols("Bet ~ PreflopV + FlopV + RiverV + FlopRV + TurnRV + RiverRV", data=df).fit()
    # print bet_size_model.summary()

    #
    # d = (0, 1000)
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['PreflopV'], d[0], d[1]), 'o', label='PreflopV')
    # plt.legend(loc=0)
    # plt.show()
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['FlopV'], d[0], d[1]), 'o', label='FlopV')
    # plt.legend(loc=0)
    # plt.show()
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['TurnV'], d[0], d[1]), 'o', label='TurnV')
    # plt.legend(loc=0)
    # plt.show()
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['RiverV'], d[0], d[1]), 'o', label='RiverV')
    # plt.legend(loc=0)
    # plt.show()
    #
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['FlopRV'], d[0], d[1]), 'o', label='FlopRV')
    # plt.legend(loc=0)
    # plt.show()
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['TurnRV'], d[0], d[1]), 'o', label='TurnRV')
    # plt.legend(loc=0)
    # plt.show()
    # plt.plot(get_range(df['Bet'], d[0], d[1]), get_range(df['RiverRV'], d[0], d[1]), 'o', label='RiverRV')
    # plt.legend(loc=0)
    # plt.show()

    # plt.plot(get_range(df['TurnV'], d[0], d[1]), get_range(df['TurnRV'], d[0], d[1]), 'o')
    # plt.show()
    # plt.plot(get_range(df['TurnV'], d[0], d[1]), get_range(df['FlopV'], d[0], d[1]), 'o')
    # plt.show()
    # plt.plot(get_range(df['TurnV'], d[0], d[1]), get_range(df['RiverV'], d[0], d[1]), 'o')
    # plt.show()