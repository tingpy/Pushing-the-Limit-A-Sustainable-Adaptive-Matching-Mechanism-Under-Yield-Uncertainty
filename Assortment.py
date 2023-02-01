import gurobipy as gp
from sklearn.linear_model import LogisticRegression
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy
import time
from itertools import compress
import random
import copy

square_spec = ['色相(調) - YI', '拉力強度 - TSy ( 50 mm/min)', '拉力強度 - TSb ( 50 mm/min)',
               '流動係數   MVR  220℃×10kg', 'Charpy Impact', '拉力強度 - EL ( 50 mm/min)']

def logistic_result(X, y):
    model = LogisticRegression(max_iter = 10000)
    model.fit(X, y)
    
    X_indep = X.groupby(X.columns.tolist(), as_index=False).size()
    pred = model.predict_proba(X_indep.drop('size', axis=1))
    X_inte = np.concatenate([np.ones((X_indep.shape[0], 1)), 
                              X_indep.drop('size', axis=1)], axis=1)
    V = np.diagflat(np.product(pred, axis=1) * X_indep['size'].values)
    
    sigma = 0
    if np.linalg.det(np.dot(np.dot(X_inte.T, V), X_inte)) == 0:
        sigma = np.linalg.pinv(np.dot(np.dot(X_inte.T, V), X_inte))
    else:
        sigma = np.linalg.inv(np.dot(np.dot(X_inte.T, V), X_inte))
        
    return np.concatenate([model.intercept_, model.coef_[0]]), sigma

# def logistic_result(X, y):
#     model = LogisticRegression(max_iter = 10000)
#     model.fit(X, y)
    
#     X_indep = X.groupby(X.columns.tolist(), as_index=False).size()
#     pred = model.predict_proba(X_indep.drop('size', axis=1))
#     X_inte = np.concatenate([np.ones((X_indep.shape[0], 1)), 
#                               X_indep.drop('size', axis=1)], axis=1)
#     V = np.diagflat(np.product(pred, axis=1) * X_indep['size'].values)
    
#     sigma = 0
#     if np.linalg.det(np.dot(np.dot(X_inte.T, V), X_inte)) == 0:
#         sigma = np.linalg.pinv(np.dot(np.dot(X_inte.T, V), X_inte))
#     else:
#         sigma = np.linalg.inv(np.dot(np.dot(X_inte.T, V), X_inte))
        
#     return np.concatenate([model.intercept_, model.coef_[0]]), sigma, \
#         np.linalg.det(np.dot(np.dot(X_inte.T, V), X_inte))
        

def generate_initial_beta(initial, cust_deal_df, cust_summary, std_dic, keep_spec, square_spec):
    group_cnt = cust_summary['group'].max()
    
    # group initial beta
    tmp_df = cust_deal_df[cust_deal_df['date'] <= initial]
    tmp_dic = dict(tuple(tmp_df.groupby('group')))
    coef_dic = {}
    total_len = len(keep_spec) + len(square_spec)
    new_col = keep_spec + [i+'^2' for i in square_spec] 
    for group in range(group_cnt):
        tmp_ind_df = pd.concat([tmp_dic[group+1][keep_spec], np.power(tmp_dic[group+1][square_spec], 2),
                                tmp_dic[group+1]['record']], axis=1)
        X = (tmp_ind_df.iloc[:,0:total_len] - std_dic['mean'][1:]) / std_dic['std'][1:] # w/o intercept
        X.columns = new_col
        y = tmp_ind_df.iloc[:,total_len]
        mu, sigma = logistic_result(X, y)
        sigma = np.identity(len(mu))
        
        coef_dic[group+1] = {'coef':{'mu': mu, 'sigma': sigma},
                             'group': group+1}
    
    # individual initial beta
    initial_dic = {}
    receiver_dic = cust_summary[['receiver', 'group']].to_dict('records')
    for item in receiver_dic:
        # if item['receiver'] == '1000171':
        #     raise ValueError
        record_df = tmp_df[tmp_df['receiver'] == item['receiver']]
        return_cnt = np.sum(record_df['record'] == -1)
        order_cnt = record_df.shape[0]
        if order_cnt == 0:
            continue
        if return_cnt/order_cnt > 0 and return_cnt/order_cnt < 1:
            X = pd.concat([record_df[keep_spec], np.power(record_df[square_spec], 2)], axis=1)
            X = (X - std_dic['mean'][1:]) / std_dic['std'][1:]
            X.columns = new_col
            y = record_df['record']
            mu, sigma = logistic_result(X, y)
            sigma = np.identity(len(mu))
            initial_dic[item['receiver']] = {'coef':{'mu': mu, 'sigma': sigma},
                                              'group': item['group']}
        else:
            initial_dic[item['receiver']] = copy.deepcopy(coef_dic[item['group']])
    
    return coef_dic, initial_dic

# def generate_initial_beta(initial, cust_deal_df, cust_summary, keep_spec, square_spec):
#     group_cnt = cust_summary['group'].max()
    
#     # group initial beta
#     tmp_df = cust_deal_df[cust_deal_df['date'] <= initial]
#     tmp_dic = dict(tuple(tmp_df.groupby('group')))
#     coef_dic = {}
#     total_len = len(keep_spec) + len(square_spec)
#     new_col = keep_spec + [i+'^2' for i in square_spec] 
#     for group in range(group_cnt):
#         print(group+1)
#         tmp_ind_df = pd.concat([tmp_dic[group+1][keep_spec], np.log(np.power(tmp_dic[group+1][square_spec], 2)),
#                                 tmp_dic[group+1]['record']], axis=1)
#         X = tmp_ind_df.iloc[:,0:total_len]
#         X.columns = new_col
#         y = tmp_ind_df.iloc[:,total_len]
#         mu, sigma = logistic_result(X, y)
        
#         coef_dic[group+1] = {'mu': mu, 'sigma': sigma}
    
#     # individual initial beta
#     initial_dic = {}
#     receiver_dic = cust_summary[['receiver', 'group']].to_dict('records')
#     for item in receiver_dic:
#         record_df = tmp_df[tmp_df['receiver'] == item['receiver']]
#         return_cnt = np.sum(record_df['record'] == -1)
#         order_cnt = record_df.shape[0]
#         if order_cnt == 0:
#             continue
#         if return_cnt/order_cnt > 0 and return_cnt/order_cnt < 1:
#             X = pd.concat([record_df[keep_spec], np.log(np.power(record_df[square_spec], 2))], axis=1)
#             X.columns = new_col
#             y = record_df['record']
#             mu, sigma = logistic_result(X, y)
#             initial_dic[item['receiver']] = {'coef':{'mu': mu, 'sigma': sigma},
#                                               'group': item['group']}
#             # ggg = np.linalg.det(sigma)
#             # print(f'''cust:{item['receiver']} group:{item['group']} det:{det_val} ggg:{ggg}''')
#         else:
#             initial_dic[item['receiver']] = {'coef': coef_dic[item['group']],
#                                               'group': item['group']}
    
#     return coef_dic, initial_dic



# def initial_current_weight_update(start, initial, cust_deal_df, spec_df):
#     order_df = cust_deal_df[np.logical_and(cust_deal_df['date'] >= start,
#         cust_deal_df['date'] <= initial)][['LOTNO', 'weight']].groupby('LOTNO').sum()
#     order_df.columns = ['current weight']

#     tmp_spec_df = spec_df.drop(['current weight'], axis=1)
#     tmp_spec_df = tmp_spec_df.merge(order_df, on='LOTNO')
    
#     return tmp_spec_df

def initial_current_weight_update(start, initial, cust_deal_df, spec_df, limit_weight):
    order_df = cust_deal_df[np.logical_and(cust_deal_df['date'] >= start,
        cust_deal_df['date'] <= initial)][['LOTNO', 'weight']].groupby('LOTNO').sum()
    order_df.columns = ['current weight']

    tmp_spec_df = spec_df #.drop(['current weight'], axis=1)
    tmp_spec_df = tmp_spec_df.merge(order_df, on='LOTNO', how='left').fillna(0)
    tmp_spec_df['budget'] = tmp_spec_df['total weight'] - tmp_spec_df['current weight']
    
    tmp_spec_df = tmp_spec_df[np.logical_and(tmp_spec_df['produce date'] >= start,
                                             tmp_spec_df['produce date'] <= initial)]
    tmp_inv_df = tmp_spec_df[['LOTNO', 'budget', 'produce date']][tmp_spec_df['budget'] > limit_weight]
    tmp_left_df = tmp_spec_df[['LOTNO', 'budget', 'produce date']][np.logical_and(tmp_spec_df['budget'] <= limit_weight,
                                                                                  tmp_spec_df['budget'] > 0)]
    
    return tmp_inv_df, tmp_left_df

# cur_spec_df = initial_current_weight_update(start, initial, cust_deal_df, spec_df)
    


# input variable matrix
def input_var_matrix(spec_df, keep_spec, square_spec, standardized):
    lot_list = list(spec_df['LOTNO'])
    var_list = ['intercept'] + keep_spec + [i+'^2' for i in square_spec]
    
    orig_var = spec_df[keep_spec].values
    orig_var_std = (orig_var - orig_var.mean(axis=0)) / orig_var.std(axis=0)
    sqr_var = np.power(spec_df[square_spec].values, 2)
    sqr_var_std = (sqr_var - sqr_var.mean(axis=0)) / sqr_var.std(axis=0)
    if standardized:
        var_mat = np.concatenate([np.ones((orig_var.shape[0], 1)), orig_var_std,
                                  sqr_var_std], axis=1)
        std_dic = {'mean': np.concatenate([np.zeros(1), orig_var.mean(axis=0), sqr_var.mean(axis=0)]),
                   'std': np.concatenate([np.ones(1), orig_var.std(axis=0), sqr_var.std(axis=0)])}
    else:
        var_mat = np.concatenate([np.ones((orig_var.shape[0],1)), orig_var, sqr_var], axis=1)
        std_dic = {'mean': np.zeros(var_mat.shape[1]), 'std': np.ones(var_mat.shape[1])}                          
    
    return {'lot':lot_list, 'var list':var_list, 'var mat':var_mat}, std_dic
# def input_var_matrix(spec_df, keep_spec, square_spec):
#     lot_list = list(spec_df['LOTNO'])
#     var_list = ['intercept'] + keep_spec + [i+'^2' for i in square_spec]
#     var_mat = np.concatenate([np.ones((spec_df.shape[0],1)), spec_df[keep_spec].values,
#                               np.log(np.power(spec_df[square_spec].values, 2))], axis=1)
    
#     return {'lot':lot_list, 'var list':var_list, 'var mat':var_mat}

# var_dic = input_var_matrix(spec_df, keep_spec, square_spec)

def cur_var_matrix(var_dic, inv_bud_df):
    ind_list = []
    total_lot = np.array(var_dic['lot'])
    for lot in inv_bud_df['LOTNO']:
        ind_list.append(np.where(total_lot == lot)[0])
    return var_dic['var mat'][np.concatenate(ind_list),:]    


def BLP_prob(S, beta_dic, var_dic):
    ret_dic = {}
    for cust in beta_dic:
        u = np.zeros(len(S))
        for k, lot in enumerate(S):
            where = var_dic['lot'].index(lot)
            u[k] = beta_dic[cust]['coef']['mu'].T @ var_dic['var mat'][where,:]
        prob = np.exp(u) / np.exp(u).sum()
        tmp_dic = {'prob': prob, 'u': u}
        ret_dic[cust] = {'param': tmp_dic, 'group': beta_dic[cust]['group']}
        
    return ret_dic

def BLP_prob_new(beta_dic, var_mat):
    ret_dic = {}
    for cust in beta_dic:
        u = var_mat @ beta_dic[cust]['coef']['mu']
        prob = np.exp(u) / np.exp(u).sum()
        ret_dic[cust] = {'param':{'prob': prob, 'u': u}, 'group': beta_dic[cust]['group']}
        
    return ret_dic
        


def new_comer_beta_update(cur_cust_df, ind_beta, weighting_beta_dic):
    tmp_df = cur_cust_df[['receiver', 'group']]
    tmp_df.drop_duplicates(subset=['receiver'], keep='first', inplace=True)
    
    for i, cust in enumerate(tmp_df['receiver']):
        if cust not in ind_beta.keys():
            ind_beta[cust] = copy.deepcopy(weighting_beta_dic[tmp_df.iloc[i,1]])
                                  
    return ind_beta


# update the weighting beta for each group and overall 
def update_group_beta(backward, end_date, cur_cust_df, beta_dic):
    start_date = end_date - timedelta(days=backward)
    
    weight_df, group_weight, total_weight = demand_update(cur_cust_df)
    
    weighting_beta_dic = {}
    total_beta_dic = {}
    for group in range(group_weight.shape[0]):
        tmp_group_df = weight_df[weight_df['group'] == group+1][['receiver', 'weight']]
        ratio_val = (tmp_group_df['weight'] / group_weight.iloc[group, 0]).values
        
        group_para = {'coef':{'mu': sum([ratio_val[i] * beta_dic[cust]['coef']['mu'] 
                                 for i, cust in enumerate(tmp_group_df['receiver'].values)]),
                      'sigma': sum([ratio_val[i]**2 * beta_dic[cust]['coef']['sigma']
                                    for i, cust in enumerate(tmp_group_df['receiver'].values)])},
                      'group': group+1}
        weighting_beta_dic[group+1] = group_para
    
    group_weight = group_weight / total_weight
    for group in range(group_weight.shape[0]):
        total_beta_dic[group+1] = {'coef':{'mu':sum([group_weight.iloc[i,0] * weighting_beta_dic[i+1]['coef']['mu'] 
                                             for i in range(group_weight.shape[0])]),
                                  'sigma':sum([group_weight.iloc[i,0]**2 * weighting_beta_dic[i+1]['coef']['sigma'] 
                                             for i in range(group_weight.shape[0])])},
                                  'group':group+1}
                
    return weighting_beta_dic, total_beta_dic

# for i, cust in enumerate(tmp_group_df['receiver'].values):
#     sum([ratio_val[i] * beta_dic[cust]['coef']['mu']])
    
# backward = 30
# end_date = MonSun[1]
# weighting_beta_dic, total_beta_dic = update_group_beta(backward, end_date, cust_deal_df, beta_dic)


# return individual demand, group demand and overall demand for a period
def demand_update(cur_cust_df):
    weight_df = cur_cust_df.groupby('receiver').agg({'group': lambda x: x.iloc[0],
                                                     'weight': 'sum'})
    weight_df['receiver'] = weight_df.index
    group_weight = weight_df.groupby('group').sum()
    total_weight = weight_df['weight'].sum()
    
    return weight_df, group_weight, total_weight

# start_date = end_date - timedelta(days=7)
# weight_df, group_weight, total_weight =  demand_update(start_date, end_date, cust_deal_df)
                           

def V_func(ind_p_dic, group_p_dic, S_all):
    total_cnt = len(S_all)
    total_group = len(group_p_dic)
    
    ret_val = np.zeros((total_cnt, total_group))
    for cust, item in ind_p_dic.items():
        ret_val[:, item['group']-1] = ret_val[:, item['group']-1] + \
            (item['param']['u'] - group_p_dic[item['group']]['param']['u']) ** 2
            
    ret_val = ret_val / np.amax(np.abs(ret_val), axis = 0)
    
    return ret_val
            

def G_func(group_p_dic, all_p_dic, S_all):
    total_cnt = len(S_all)
    total_group = len(group_p_dic)
    
    ret_val = np.zeros((total_cnt, total_group))
    for group in group_p_dic:
        ret_val[:,group-1] = group_p_dic[group]['param']['prob'] - \
            all_p_dic[group]['param']['prob']
            
    ret_val = ret_val / np.amax(np.abs(ret_val), axis = 0)
    
    return ret_val

def u_bar_func(G_val, V_val, alpha):
    orig_u_bar = alpha*G_val + (1-alpha)*V_val
    u_bar = orig_u_bar - orig_u_bar.min(axis=0) / (orig_u_bar.max(axis=0) - orig_u_bar.min(axis=0))
    
    return u_bar


def U_func(u_bar, pos_dic, alpha):
    big_U = {c:{} for c in pos_dic}
    # euler_const = 0.5772156649
    euler_const = 0
    
    for group, item in pos_dic.items():
        for s, combination in item.items():
            tmp_val = u_bar[combination, group-1]
            tmp_val_exp = np.exp(tmp_val)
            big_U[group][s] = np.log(np.sum(tmp_val_exp)) + euler_const
    
    return big_U


def position_indicator(S0, S_all):
    pos_dic = {}
    for c, item in S0.items():
        tmp_dic = {}
        for j, lot_list in item.items():
            tmp_dic[j] = np.concatenate([np.where(S_all == lotno)[0] for lotno in lot_list])
        pos_dic[c] = tmp_dic
        
    return pos_dic

# pos = position_indicator(S0, S_all)


def cur_cust_spec_df(cur_date, end_date, cust_deal_df, inv_bud_df, leftover_df, spec_df, limit_weight):
    tmp_cust_df = cust_deal_df[np.logical_and(cust_deal_df['date'] >= cur_date,
                                              cust_deal_df['date'] <= end_date)]
    
    new_produce = spec_df[np.logical_and(spec_df['produce date'] >= cur_date,
                                         spec_df['produce date'] <= end_date)]
    new_produce['budget'] = new_produce['total weight']
    tmp_inv_df = pd.concat([copy.deepcopy(inv_bud_df), new_produce[['LOTNO', 'budget', 'produce date']]], axis=0)
    
    tmp_left_df = pd.concat([copy.deepcopy(leftover_df), tmp_inv_df[np.logical_and(tmp_inv_df['budget'] <= limit_weight,
                                                                    tmp_inv_df['budget'] > 0)]], axis=0)
    tmp_inv_df = tmp_inv_df[tmp_inv_df['budget'] > limit_weight]
    S = tmp_inv_df['LOTNO'].unique()
    
    return tmp_cust_df, tmp_inv_df, S, tmp_left_df


def initial_candidate(cur_cust_df, inv_bud_df, return_dic, u_bar, weight_limit, pre_ratio, seed, combination_limit):
    np.random.seed(seed)
    
    group_dic = dict(tuple(cur_cust_df.groupby('group')))
    combination_limit = min(inv_bud_df.shape[0], combination_limit)
    
    S0_c = {c:{} for c in group_dic}
    avg_cnt = []
    iter_limit = min(100, inv_bud_df.shape[0])
    # demand_dic = {c:{} for c in group_dic}
    logic_vec1 = inv_bud_df['budget'] >= weight_limit
    for c, item in group_dic.items():
        logic_vec2 = [i not in return_dic[c] for i in inv_bud_df['LOTNO']]
        logic_vec = [a and b for a, b in zip(logic_vec1, logic_vec2)]
        tmp_spec_df_c = inv_bud_df[logic_vec]
        new_u_bar = u_bar[logic_vec,]
        spec_weight = tmp_spec_df_c['budget'].values
        S = tmp_spec_df_c['LOTNO'].values
        cnt_vec = []
        
        group_order = item['weight'].sum() * pre_ratio[c-1]

        combination_cnt = 0
        weight_rank = np.argsort(spec_weight)
        start_index = np.random.choice(tmp_spec_df_c.shape[0], 
                                       min(iter_limit, tmp_spec_df_c.shape[0]), replace=False)
        cnt_limit = 5
        iter_cnt = 0
        while combination_cnt < combination_limit and iter_cnt < iter_limit:
            cand_list = []
            for i in np.concatenate([weight_rank[start_index[iter_cnt]:], weight_rank[:start_index[iter_cnt]]]):
                cand_list.append(i)
                cur_prob = np.exp(new_u_bar[cand_list,c-1]) / np.sum(np.exp(new_u_bar[cand_list,c-1]))
                if len(cand_list) >= cnt_limit:
                    logic_vec = group_order * cur_prob < spec_weight[cand_list]
                    if np.all(logic_vec):
                        S0_c[c][combination_cnt] = cand_list 
                    else:
                        tmp_list = list(compress(cand_list, logic_vec))
                        tmp_logic = group_order * np.exp(new_u_bar[tmp_list,c-1]) / np.sum(np.exp(new_u_bar[tmp_list,c-1])) < spec_weight[tmp_list]
                        if ~(len(cand_list) - len(tmp_list) < cnt_limit and np.all(tmp_logic)):
                            continue
                        cand_list = tmp_list
                    S0_c[c][combination_cnt] = S[cand_list]
                    cnt_vec.append(len(cand_list))
                    combination_cnt = combination_cnt + 1
                    cand_list = []
                    break
            iter_cnt += 1
        avg_cnt.append(np.mean(cnt_vec))
    
    return S0_c, avg_cnt


# def demand_for_lotno(group_weight, pre_ratio, S0, u_bar, pos_dic):
#     demand_dic = {c:{} for c in pos_dic}
#     all_cnt = u_bar.shape[0]
    
#     for c in range(u_bar.shape[1]):
#         tmp_prob = u_bar[:,c]
#         for s, loc in pos_dic[c+1].items():
#             tmp_vec = np.zeros(all_cnt)
#             tmp_vec[loc] = group_weight.iloc[c,0] * tmp_prob[loc] / tmp_prob[loc].sum() * pre_ratio
#             demand_dic[c+1][s] = tmp_vec

#     return demand_dic 


def demand_for_lotno(group_weight, pre_ratio, S0, u_bar, pos_dic):
    demand_dic = {c:{} for c in pos_dic}
    all_cnt = u_bar.shape[0]
    
    for c in S0:
        tmp_prob = u_bar[:,c-1]
        for s, loc in pos_dic[c].items():
            tmp_vec = np.zeros(all_cnt)
            tmp_vec[loc] = group_weight.iloc[c-1,0] * tmp_prob[loc] / tmp_prob[loc].sum() * pre_ratio[c-1]
            demand_dic[c][s] = tmp_vec

    return demand_dic 

def return_func(group_p_dic, S_all, cutoff):
    ret_dic = {c:{} for c in group_p_dic}
    
    for c, item in group_p_dic.items():
        tmp_val = 1 / (1 + np.exp(item['param']['u']))
        ret_dic[c] = {'ret prob': tmp_val, 'ret': tmp_val > cutoff}
        
    return ret_dic


# # construct primal problem
# def construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic,
#                      S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight, pre_ratio,
#                      alpha, E1, E2, E3, verbose, seed):
    
#     # primal model parameters
#     m = gp.Model("Assortment Primal")
#     m.ModelSense = -1 ## maximization problem
#     if not verbose:
#         m.setParam('OutputFlag', False)
    
#     # demand
#     w = group_weight / total_weight
    
#     # G_c(S) and V_c(S)
#     G_val = G_func(group_p_dic, all_p_dic, S_all)
#     V_val = V_func(ind_p_dic, group_p_dic, S_all)
#     u_bar = u_bar_func(G_val, V_val, alpha)
    
#     # red line inventory
#     return_dic = return_func(group_p_dic, S_all, E3)
    
#     # initial S
#     S0, E2 = initial_candidate(cur_cust_df, inv_bud_df, return_dic, u_bar, limit_weight, pre_ratio, seed, combination_limit=30)
#     pos_dic = position_indicator(S0, S_all)
#     lot_demand = demand_for_lotno(group_weight, pre_ratio, S0, u_bar, pos_dic) ###
#     U_val = U_func(u_bar, pos_dic, alpha)
    
    
#     # decision variable
#     x = {(c+1):{} for c in range(len(S0)) if len(S0[c+1]) > 0}
#     for c in S0:
#         for j in range(len(S0[c])):
#             x[c][j] = m.addVar(lb = 0, obj = w.iloc[c-1,0] * U_val[c][j])
#     m.update()   
    
#     capacityConstr = []
#     for k in range(inv_bud_df.shape[0]):
#         capacityExpr = gp.quicksum(x[c][j] * lot_demand[c][j][k] for c in S0 for j in lot_demand[c])
#         capacityConstr.append(m.addConstr(capacityExpr <= inv_bud_df.iloc[k,1], name='capacity_%d' % k))
#         # capacityConstr.append(m.addConstr(capacityExpr <= 100000, name='capacity_%d' % k))
    
#     regularConstr = {}
#     for c in S0:
#         regularConstr[c] = m.addConstr(gp.quicksum(x[c][j] for j in range(len(S0[c]))) == 1,
#                                        name='regular_%r' % c)
        
#     cardinalityConstr = {}
#     for c in S0:
#         cardinalityConstr[c] = m.addConstr(gp.quicksum(x[c][j] * len(S0[c][j]) for j in S0[c]) <= E2[c-1], 
#                                             name='cardinality_%r' % c)
#         # cardinalityConstr[c] = m.addConstr(gp.quicksum(x[c][j] * len(S0[c][j]) for j in range(len(S0[c]))) <= 20, 
#         #                                    name='cardinality_%r' % c)
    
#     m.update()
#     constraints = [capacityConstr, regularConstr, cardinalityConstr]   
    
#     return m, x, constraints, u_bar, return_dic, lot_demand, S0, E2

def construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic,
                      S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight, pre_ratio, del_num,
                      alpha, E1, E2, E3, verbose, seed):
    
    # primal model parameters
    m = gp.Model("Assortment Primal")
    m.ModelSense = -1 ## maximization problem
    if not verbose:
        m.setParam('OutputFlag', False)
    
    # demand
    w = group_weight / total_weight
    
    # G_c(S) and V_c(S)
    G_val = G_func(group_p_dic, all_p_dic, S_all)
    V_val = V_func(ind_p_dic, group_p_dic, S_all)
    u_bar = u_bar_func(G_val, V_val, alpha)
    
    # red line inventory
    return_dic = return_func(group_p_dic, S_all, E3)
    
    # initial S
    S0, E2 = initial_candidate(cur_cust_df, inv_bud_df, return_dic, u_bar, limit_weight, pre_ratio, seed, combination_limit=30)
    pos_dic = position_indicator(S0, S_all)
    lot_demand = demand_for_lotno(group_weight, pre_ratio, S0, u_bar, pos_dic) ###
    U_val = U_func(u_bar, pos_dic, alpha)
    
    
    # decision variable
    x = {c:{} for c in S0 if len(S0[c]) > 0}
    for c in S0:
        for j in range(len(S0[c])):
            x[c][j] = m.addVar(lb = 0, obj = w.iloc[c-1,0] * U_val[c][j])
    m.update()   
    
    capacityConstr = []
    for k in range(inv_bud_df.shape[0]):
        capacityExpr = gp.quicksum(x[c][j] * lot_demand[c][j][k] for c in S0 for j in lot_demand[c])
        capacityConstr.append(m.addConstr(capacityExpr <= inv_bud_df.iloc[k,1], name='capacity_%d' % k))
        # capacityConstr.append(m.addConstr(capacityExpr <= 100000, name='capacity_%d' % k))
    
    regularConstr = {}
    for c in S0:
        if c == del_num:
            regularConstr[c] = m.addConstr(gp.quicksum(x[c][j] for j in range(len(S0[c]))) == 0,
                                            name='regular_%r' % c)
        else:
            regularConstr[c] = m.addConstr(gp.quicksum(x[c][j] for j in range(len(S0[c]))) == 1,
                                            name='regular_%r' % c)
            
    cardinalityConstr = {}
    for c in S0:
        cardinalityConstr[c] = m.addConstr(gp.quicksum(x[c][j] * len(S0[c][j]) for j in S0[c]) <= E2[c-1], 
                                            name='cardinality_%r' % c)
        # cardinalityConstr[c] = m.addConstr(gp.quicksum(x[c][j] * len(S0[c][j]) for j in range(len(S0[c]))) <= 20, 
        #                                    name='cardinality_%r' % c)
    
    m.update()
    constraints = [capacityConstr, regularConstr, cardinalityConstr]   
    
    return m, x, constraints, u_bar, return_dic, lot_demand, S0, E2



### remove mandatory
def OptimalAssortment(n, v, r, cost, allowed, limit, a, b, verbose=False):
    v = np.array(v)
    r = np.array(r)
    limit=np.array(limit)
    vr = v * r
    optionalList = np.array(allowed)
    limitList = [i for i in optionalList if limit[i]]

    tmp = sorted([(v[i], r[i], -i) for i in limitList], reverse=True)
    jOrder = [-entry[-1] for entry in tmp]
    o = np.zeros(n, dtype=int)
    kmax = limit[list(allowed)].sum()
    kmin = 0
    vsum = np.zeros(kmax + 1)
    vrsum = np.zeros(kmax + 1)
    
    initialM = np.zeros(n, dtype=bool)

    M = {}
    M[kmin] = np.array(initialM)
    optZ = 0
    optM = initialM

    def update(optZ, optM, k, vsum, vrsum, M):
        # if vsum[k] < 0:
        #     raise ValueError("invalid log")
        z = a * np.log(vsum[k]) + b * vrsum[k] / vsum[k] - cost[k]
        if verbose:
            print(f'COMPUTED z={z} optZ={optZ}')    
        if z > optZ:
            if verbose:
                print(f'UPDATING using k={k} optZ={z} ')
            return z, np.array(M[k])
        return optZ, optM
    
    indices = np.arange(n)
    # optZ, optM = update(optZ, optM, kmin, vsum, vrsum, M)
    for i, j in enumerate(jOrder):
        o[j] = i + 1
        k = kmin + i + 1
        vsum[k] = vsum[k - 1] + v[j]
        vrsum[k] = vrsum[k - 1] + vr[j]
        M[k] = np.array(M[k - 1])
        M[k][j] = 1
        optZ, optM = update(optZ, optM, k, vsum, vrsum, M)
    tau = sorted([((vr[i] - vr[j]) / (v[i] - v[j]), -i, j) for i in limitList for j in limitList if v[i] > v[j]] + [(r[i], -i, n) for i in optionalList])
    
    for unused, minusI, j in tau:
        i = -minusI
        # if verbose:
        #    print('\no=', o, '\nvsum=', vsum, '\nvrsum=', vrsum, '\nM=', {k:indices[M[k]] for k in M}, '\n')
        #    print(f'lamda={unused} i={i} j={j}')
            
        if j == n:
            for k in range(kmin, kmax + 1):
                if M[k][i]:
                    if verbose:
                        print(f'Taking out i={i} with o[i]={o[i]}')
                    vsum[k] -= v[i]
                    vrsum[k] -= vr[i]
                    M[k][i] = 0
                    optZ, optM = update(optZ, optM, k, vsum, vrsum, M)
        elif o[i] < o[j]:
            o[i], o[j] = o[j], o[i]
            k = o[j] + kmin
            if M[k][i]:
                if verbose:
                    print(f'Swapping out i={i} with o[i]={o[i]} with j={j} and o[j]={o[j]}')
                vsum[k] += v[j] - v[i]
                vrsum[k] += vr[j] - vr[i]
                M[k][j] = 1
                M[k][i] = 0
                optZ, optM = update(optZ, optM, k, vsum, vrsum, M)
    # if verbose:
    #     print('\no=', o, '\nvsum=', vsum, '\nvrsum=', vrsum, '\nM=', {k:indices[M[k]] for k in M}, '\n')
    
    return optZ, indices[optM]


# def test_dual(n, v, r, a, b, cost, pos_dic):
#     v = np.array(v)
#     r = np.array(r)
#     vr = v * r
    
#     val_dic = {c:[] for c in pos_dic}
#     for c, item in pos_dic.items():
#         for item1 in item.values():
#             val_dic[c].append(a*np.log(np.sum(v[item1])) + b*np.sum(vr[item1])/np.sum(v[item1]) - cost[len(item1)])
            
#     print([np.max(val_dic[c]) for c in pos_dic])

# test_dual(n, v, r, a, b, cost, pos_dic)


### allowed = all True if not reoptimize
def group_assortment(S0, inv_bud_df, group_weight, return_dic, allowed, bad_list, u_bar, pre_ratio,
                     lamda, mu, xi, x, limit_weight, thres):
    new_mu = np.zeros(len(S0))
    new_cand_dic = {}
    total_weight =  group_weight.iloc[:,0].sum()
    
    for c in S0:
        if c in bad_list:
            new_mu[c-1] = mu[c-1]
            continue
        if group_weight.iloc[c-1,0] > 0:
            n = inv_bud_df.shape[0]
            v = np.exp(u_bar[:,c-1])
            r = -lamda
            # cost = xi[c-1] * np.array([k+1 if k < k_thres[c-1] else 1e20 for k in range(inv_bud_df.shape[0])])
            limit = ~return_dic[c]['ret']
            cost = xi[c-1] * np.array([k if k <= limit.sum() else 1e20 for k in range(inv_bud_df.shape[0]+1)])
            
            a = group_weight.iloc[c-1,0] / total_weight
            # b = group_weight.iloc[c-1,0] / total_weight
            b = group_weight.iloc[c-1,0] * pre_ratio[c-1]
            allowed = []
            if len(allowed) == 0:
                logic_vec = [a and b for a, b in zip(inv_bud_df['budget'] > limit_weight, ~return_dic[c]['ret'])]
                allowed = [i for i, item in enumerate(logic_vec) if item]

            new_mu[c-1], opt_cand = OptimalAssortment(n, v, r, cost, allowed, limit, a, b, verbose=False)
            
            if new_mu[c-1] <  mu[c-1] - thres:
                totalCost = -r
                #origZ, origS = extractPolicy()
                print('Error dump:')
                print('group ', c)
                print('orignal value', mu[c-1])
                print('new value', new_mu[c-1])          
                print('a= %s \t b=%s \t xi=%s' % (a, b, xi))
                
                # print("ValueError( Do not meet the original! )")
                bad_list.append(c)
                new_mu[c-1] = mu[c-1]
                # new_cand_dic[c] = opt_cand
                global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = \
                    f'''Subproblem did not meet original, cluster {c}'''
                # raise ValueError('Subproblem did not meet original!')
            if new_mu[c-1] > mu[c-1]:
                if len(opt_cand) > 0:
                    new_cand_dic[c] = opt_cand
    
    return new_mu, new_cand_dic, bad_list

            
def ExtractShadowPrices(constr):
    if type(constr) == list:
        return np.array([c.Pi for c in constr])
    elif type(constr) == dict:
        return np.array([constr[i].Pi if constr[i] else 0 for i in constr])
    
 
def dualObj_func(mu, lamda, xi, inv_bud_df, E2):
    return np.sum(mu) + np.dot(lamda, inv_bud_df['budget']) + np.sum(xi*E2)


def update_candidate(S_all, S_index, old_S):
    for c in old_S:
        total_cnt = len(old_S[c])
        if c in S_index.keys():
            old_S[c][total_cnt] = S_all[S_index[c]]
    
    return old_S


def addColumns(m, x, constraints, S_index, group_weight, u_bar, pre_ratio):
    # euler_const = 0.5772156649
    euler_const = 0
    capacityConstr, regularConstr, cardinalityConstr = constraints
    w = group_weight / group_weight.iloc[:,0].sum()
    
    for c, M in S_index.items():
        x_c = x[c]
        if len(M) > 0 and group_weight.iloc[c-1,0] > 0:
            k = len(x_c)
            # for M in S_index[c]:
                ## v is big U, p is BLP prob
            p = np.exp(u_bar[M,c-1]) / np.exp(u_bar[M,c-1]).sum()
            v = np.log(np.exp(u_bar[M,c-1]).sum()) + euler_const
            limit_cnt = len(S_index[c])
            
            col = gp.Column()
            
            col.addTerms(p * group_weight.iloc[c-1,0] * pre_ratio[c-1], [capacityConstr[s] for s in M])
            col.addTerms(1, regularConstr[c])
            col.addTerms(limit_cnt, cardinalityConstr[c])
            x_c[k] = m.addVar(lb=0, obj=w.iloc[c-1,0] * v, column=col)
            k += 1
    m.update()


def extractPolicy(x, new_S):
    final_S = {c:{} for c in new_S}
    final_p = {c:[] for c in new_S}
    for c, item in x.items():
        com_cnt = 0
        for s, val in item.items():
            if val.X > 0:
                final_S[c][s] = new_S[c][s]
                final_p[c].append(val.X)
                com_cnt = com_cnt + 1
                
    return final_S, final_p


def quota_and_dist_update(S_all, final_pos_dic, final_p, u_bar, group_weight, pre_ratio):
    total_quota = np.zeros(len(S_all))
    quota_by_c = {c:np.zeros(len(S_all)) for c in final_pos_dic}
    
    u_exp = np.exp(u_bar)
    for c, item in final_pos_dic.items():
        for s, loc in enumerate(item.values()):
            tmp_val = final_p[c][s] * u_exp[loc, c-1] / u_exp[loc, c-1].sum() * group_weight.iloc[c-1,0] * pre_ratio[c-1]
            quota_by_c[c][loc] += tmp_val
        total_quota += quota_by_c[c]
        
    return total_quota, quota_by_c

            

# start_date = MonSun[0]
# end_date = MonSun[1]
# limit_weight = 2000
# alpha = 0.5
# thres = 1e-5

#### revision needed           
def solve(cur_cust_df, inv_bud_df, S_all, var_dic,
          ind_p_dic, group_p_dic, all_p_dic, limit_weight, pre_ratio, seed,
          alpha, E1, E2, E3, thres, iter_limit):
    # demand
    weight_df, group_weight, total_weight = demand_update(cur_cust_df)
    
    if total_weight / inv_bud_df['budget'].sum() > 0.9:
        pre_ratio = [1.02 for c in range(len(group_p_dic))]
        E3 = 1
    for c in range(group_weight.shape[0]):
        if group_weight.iloc[c,0] * (pre_ratio[c] - 1) < 500:
            pre_ratio[c] = 1 + 500 / group_weight.iloc[c,0]
    
    del_option = 0
    m, x, constraints, u_bar, return_dic, lot_demand, S0, E2 = construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic, 
                                                                S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight,
                                                                pre_ratio, del_option, alpha, E1, E2, E3, verbose = True, seed = seed)
    dualObj = np.Inf
    startT = endT = time.time()
    new_S = S0
    iter_cnt = 0
    bad_list = []
    while True:
        if verbose:
            print('*******************Solving')
        m.optimize()
        
        if verbose:
            print('*******************Done Solving')
        if m.status != gp.GRB.status.OPTIMAL:
            # m.write('C:\\Users\\eric8\\Desktop\\graduate thesis reference\\Numerical\\gurobi_record\\Infeasible.lp')
            print(f''' ValueError('Gurobi not optimal, status {m.status}')''')
            
            del_opt_list = []
            for del_num in new_S:
                m, x, constraints, u_bar, return_dic, lot_demand, S0, E2 = construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic, 
                                                                             S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight,
                                                                             pre_ratio, del_num, alpha, E1, E2, E3, verbose = True, seed = seed)
                m.optimize()
                if m.status == gp.GRB.status.OPTIMAL:
                    del_opt_list.append(del_num)
            if len(del_opt_list) == 0:
                # raise ValueError('hopeless, impossible to be feasible')
                return 'Infeasible', [], [], []  
            del_option = del_opt_list[np.argmin(group_weight.iloc[[g-1 for g in del_opt_list],0])]
            
            m, x, constraints, u_bar, return_dic, lot_demand, S0, E2 = construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic, 
                                                                         S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight,
                                                                         pre_ratio, del_option, alpha, E1, E2, E3, verbose = True, seed = seed)
            m.optimize()
            global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = f'{del_option} will be removed due to infeasibility'
            
        # if verbose:
        #     m.write('C:\\Users\\eric8\\Desktop\\graduate thesis reference\\Numerical\\gurobi_record\\mostRecent.lp')
        #     m.write('C:\\Users\\eric8\\Desktop\\graduate thesis reference\\Numerical\\gurobi_record\\mostRecent.sol')
        primalObj = m.ObjVal
        lamda, mu, xi = [ExtractShadowPrices(constr) for constr in constraints]
        # if ~isinstance(del_opt_list, list):
        #     mu[del_opt_list-1] = 0
        allowed = []
        mu2, S2_index, bad_list = group_assortment(S0, inv_bud_df, group_weight, return_dic, allowed, bad_list, u_bar, pre_ratio,
                                         lamda, mu, xi, x, limit_weight, thres)
        if del_option != 0:
            mu2[del_option-1] = 0
        dualObj = min(dualObj, dualObj_func(mu2, lamda, xi, inv_bud_df, E2))
        
        if verbose:
            print('primal Obj: ', primalObj, '  dual Obj: ', dualObj)
            print('Time elapsed ')
            print('\t\t since last cycle %.3f' % (time.time() - endT))
            print('\t\t since beginning %.3f' % (time.time() - startT))
            lastTime = time.time()
        
        if not primalObj < dualObj + thres:
            print('Primal > Dual! Something is wrong')
            print('Weights ', (group_weight/total_weight).values)
            dual_info(mu, lamda, xi)
            print('\t new feasibility (mu):%s' % mu2)
            
            print('Primal Objective %s:' % primalObj)
            print('Old dual Objective %s:' % dualObj_func(mu, lamda, xi, inv_bud_df, E2))
            print('Current dual Objective %s:' % dualObj_func(mu2, lamda, xi, inv_bud_df, E2))
            print('best dual Objective %s:' % dualObj_func)
            raise ValueError('Primal > Dual! Something is wrong!')
        if dualObj - thres <= primalObj:
            break
        iter_cnt += 1
        if iter_cnt > iter_limit:
            if dualObj - primalObj < 1e-3:
                global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = \
                    f'''ValueError(Exceed maximal iteration limit, the gap is {dualObj - primalObj})'''
                break
                # raise ValueError("Exceed maximal iteration limit, the gap is {dualObj - primalObj}")
        addColumns(m, x, constraints, S2_index, group_weight, u_bar, pre_ratio)
        new_S = update_candidate(S_all, S2_index, new_S)
    
    final_S, final_p = extractPolicy(x, new_S)
    final_pos_dic = position_indicator(final_S, S_all)
    total_quota, quota_by_c = quota_and_dist_update(S_all, final_pos_dic, final_p, u_bar, 
                                                    group_weight, pre_ratio)
    
    return total_quota, quota_by_c, del_option, u_bar

def U_func_infeasible(u_bar, alpha):
    big_U = {c:{} for c in range(u_bar.shape[0])}
    # euler_const = 0.5772156649
    euler_const = 0
    
    for group, item in pos_dic.items():
        for s, combination in item.items():
            tmp_val = u_bar[combination, group-1]
            tmp_val_exp = np.exp(tmp_val)
            big_U[group][s] = np.log(np.sum(tmp_val_exp)) + euler_const
    
    return big_U

def infeasible_solve(cur_cust_df, total_inv_df, S_all, ind_p_dic, group_p_dic, all_p_dic, alpha):
    weight_df, group_weight, total_weight = demand_update(cur_cust_df)
    
    m = gp.Model("Assortment Primal infeasible")
    m.ModelSense = -1 ## maximization problem
    if not verbose:
        m.setParam('OutputFlag', False)
    
    # demand
    w = group_weight / total_weight
    
    # G_c(S) and V_c(S)
    G_val = G_func(group_p_dic, all_p_dic, S_all)
    V_val = V_func(ind_p_dic, group_p_dic, S_all)
    u_bar = u_bar_func(G_val, V_val, alpha)
    
    # decision variable
    x = {c:{} for c in group_p_dic.keys()}
    for c in group_p_dic.keys():
        for j in range(len(S_all)):
            x[c][j] = m.addVar(lb = 0, obj = w.iloc[c-1,0] * u_bar[j,c-1])
    m.update()   
    
    capacityConstr = []
    for j in range(len(S_all)):
        capacityExpr = gp.quicksum(x[c][j] for c in group_p_dic.keys())
        capacityConstr.append(m.addConstr(capacityExpr <= total_inv_df.iloc[j,1], name='capacity_%d' % j))
        
    demandConstr = []
    for c in group_p_dic.keys():
        # q = group_weight.iloc[c-1,0] * 1.02 - (group_weight.iloc[c-1,0] * 1.02 % 100)
        demandExpr = gp.quicksum(x[c][j] for j in range(len(S_all)))
        demandConstr.append(m.addConstr(demandExpr == group_weight.iloc[c-1,0], name='demand_%d' % c))
    
    m.update()
    m.optimize()
    
    final_all = {c:{} for c in group_p_dic.keys()}
    for c, item in x.items():
        tmp_vec = []
        for j, var in item.items():
            tmp_vec.append(var.X)
        final_all[c] = np.array(tmp_vec)
    
    return final_all
        
    
    
    
    
# for del_num in range(1,11):
#     m, x, constraints, u_bar, return_dic, lot_demand, S0, E2 = test_construct_primal(ind_p_dic, group_p_dic, all_p_dic, var_dic, 
#                                                                  S_all, cur_cust_df, inv_bud_df, group_weight, total_weight, limit_weight,
#                                                                  pre_ratio, del_num, alpha, E1, E2, E3, verbose = True, seed = seed)
#     print(del_num)
#     m.optimize()



   

















