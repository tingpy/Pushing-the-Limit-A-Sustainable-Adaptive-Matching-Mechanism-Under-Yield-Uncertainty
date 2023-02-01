
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 18:03:36 2022

@author: acer
"""

import pandas as pd
import numpy as np
import scipy
import gurobipy as gp

def estimate_offline_beta(start, end, cust_deal_df, ind_initial_dic, std_dic, keep_spec, square_spec):
    # individual offline beta
    tmp_df = cust_deal_df[np.logical_and(cust_deal_df['date'] <= end,
                                          cust_deal_df['date'] >= start)]
    cust_dic = dict(tuple(tmp_df.groupby('receiver')))
    final_dic = {}
    for cust, item in cust_dic.items():
        tmp_item = item[keep_spec]
        tmp_react = item['record'].replace(-1, 0).values
        
        X = pd.concat([tmp_item, np.power(tmp_item[square_spec], 2)], axis=1)
        X = (X - std_dic['mean'][1:]) / std_dic['std'][1:]
        X.columns = keep_spec + [i+'^2' for i in square_spec]
        y = item['record'].values
        
        order_cnt = len(y)
        return_cnt = (y == -1).sum()
        
        if return_cnt/order_cnt > 0 and return_cnt/order_cnt < 1:
            mu, sigma = logistic_result(X, y)
            final_dic[cust] = {'mu': mu, 'sigma': sigma}
        else:
            X = np.concatenate([np.ones((X.shape[0],1)), X.values], axis=1)
            # X = (X - std_dic['mean']) / std_dic['std']
            mu = np.concatenate([np.ones(1)*10, np.zeros(X.shape[1]-1)])
            sigma = np.identity(X.shape[1])
    
            # for i in range(X.shape[0]):
            #     if np.linalg.det(sigma) == 0:
            #         break
            #     mu, sigma = beta_update(X[i,:], mu, sigma, 1)
            final_dic[cust] = {'mu': mu, 'sigma': sigma}
    
    return final_dic

def beta_update(quality, mu, sigma, react):
    row = len(quality)
    sigma_dec = np.linalg.cholesky(sigma)
    
    basis = np.linalg.qr(np.reshape(sigma_dec.T @ quality, (row,1)), mode = 'complete')[0]
    mu_norm = np.inner(mu, quality)
    sigma_norm = np.linalg.norm(sigma_dec.T @ quality)
    
    z_exp, z_var = update_z(mu_norm, sigma_norm, 'both') if react == 1 \
                        else update_z(mu_norm*(-1), sigma_norm, 'both') 
    
    tmp = np.identity(row)
    tmp[0,0] = z_var
    new_mu = mu + (sigma_dec.T @ basis)[:,0] * z_exp
    new_cov = (sigma_dec @ basis) @ tmp @ (sigma_dec @ basis).T
    
    return new_mu, new_cov


def update_z(m, v, ret):
    pi = np.pi
    const_func = lambda x: (1+np.exp(-m-v*x))**(-1) * np.exp(-x**2/2) / np.sqrt(2*pi)
    exp_func = lambda x: x * (1+np.exp(-m-v*x))**(-1) * np.exp(-x**2/2) / np.sqrt(2*pi)
    var_func = lambda x: x**2 * (1+np.exp(-m-v*x))**(-1) * np.exp(-x**2/2) / np.sqrt(2*pi)
    
    const = scipy.integrate.quad(const_func, -1*np.inf, np.inf)[0]
    z_exp = scipy.integrate.quad(exp_func, -1*np.inf, np.inf)[0] / const
    z_var = scipy.integrate.quad(var_func, -1*np.inf, np.inf)[0] / const - z_exp**2
    
    if ret == 'both':
        return z_exp, z_var
    elif ret == 'v':
        return const*np.sqrt(z_var)


def rankmin(x):
    u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    csum = np.zeros_like(counts)
    csum[1:] = counts[:-1].cumsum()
    
    return csum[inv]


def round_to_mini(q_list, mini):
    return np.array(q_list) - np.array(q_list) % mini


def individual_order(cur_cust_df, group_specify):
    tmp_df = cur_cust_df
    if group_specify >= 0:
        tmp_df = tmp_df[tmp_df['group'] == group_specify]
    tmp_df = tmp_df.groupby('receiver').agg({'receiver': lambda x: x.iloc[0],
                                             'weight': 'sum'})
    tmp_df = tmp_df[tmp_df['weight'] > 0]
    
    return tmp_df


def pre_allocation(ind_p_dic, quota_list, ind_order_df, mini):
    # return prob 
    ret_prob = {}
    for k in ind_order_df['receiver']:
        ret_prob[k] = 1/(1+np.exp(ind_p_dic[k]['param']['u']))
    
    # optimization model
    m_all = gp.Model("Prior Allocation")
    m_all.ModelSense = 1
    
    y = {k:{} for k in ind_order_df['receiver']}
    for k in ind_order_df['receiver']:
        for j in range(len(quota_list)):
            y[k][j] = m_all.addVar(lb = 0, obj = ret_prob[k][j])

    m_all.update()   
    
    int_quota_list = round_to_mini(quota_list, mini)
    capacityConstr = []
    for j, q in enumerate(int_quota_list):
        capacityExpr = gp.quicksum(y[k][j] for k in ind_order_df['receiver'])
        capacityConstr.append(m_all.addConstr(capacityExpr <=  q, name='capacity_%d' % j))

    demandConstr = []
    for (k, d) in zip(ind_order_df['receiver'], ind_order_df['weight']):
        demandExpr = gp.quicksum(y[k][j] for j in range(len(quota_list)))
        demandConstr.append(m_all.addConstr(demandExpr == d, name='demand_'+ k))
    
    m_all.update()
    m_all.optimize()
    
    final_all = {k:{} for k in ind_order_df['receiver']}
    # if m_all.status != gp.GRB.status.OPTIMAL:
    #     ee
    # else:
    for k, item in y.items():
        for j, var in item.items():
            if var.X > 0:
                final_all[k][j] = var.X
                
    return final_all


def trial_assign(ind_p_dic, ind_beta_dic, cur_var_mat, quota_list, prior_all, ind_order_df, mini):#, delta_list):
    uni_cust = ind_order_df['receiver']
    ava_list = np.where(quota_list > 0)[0]
    
    param_dic = {k:{} for k in uni_cust}
    for ind, k in enumerate(uni_cust):
        # mu and sigma norm
        mu = ind_beta_dic[k]['coef']['mu']
        sigma = ind_beta_dic[k]['coef']['sigma']
        
        sigma_norm = []
        for loc in ava_list:
            try:
                sigma_dec = np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError as err:
                if 'positive definite' in str(err):
                    sigma_dec = np.linalg.cholesky(sigma + 1e-1*np.identity(sigma.shape[0]))
                else:
                    raise ValueError("matrix is not p.d., fail to find a chol_decom")
            sigma_norm.append(np.linalg.norm(sigma_dec @ cur_var_mat[loc,], axis=0))
        
        mu_norm = cur_var_mat[ava_list,] @ mu
        
        # return probability
        ret_prob = 1/(1+np.exp(ind_p_dic[k]['param']['u']))
        
        # g value w.r.t each possible stock
        g = [update_z(mu_norm[loc], sigma_norm[loc], 'v') + \
             update_z(-1*mu_norm[loc], sigma_norm[loc], 'v') for loc in range(len(ava_list))]
        nan_loc = np.where(np.isnan(g))[0]
        if len(nan_loc) > 0:
            for loc in nan_loc:
                g[loc] = 1
        
        # possible change
        # admis_set = (quota_list[ava_list] > ind_order_df['weight'].iloc[ind])
        # for loc in prior_all[k].keys():
        #     admis_set[np.where(ava_list == loc)[0]] = False 
        
        param_dic[k] = {'ret': ret_prob[ava_list], 'g': g, 'admis': ava_list}
    
    m_trial = gp.Model("Trial Assignment Constrained")
    m_trial.ModelSense = -1
    
    # objective (w/o interaction)
    y = {k:{} for k in uni_cust}
    for k in uni_cust:
        for j, ind in enumerate(param_dic[k]['admis']):
            # y[k][ind] = m_trial.addVar(vtype=gp.GRB.BINARY, obj=(1-param_dic[k]['ret'][j])*param_dic[k]['g'][j])
            y[k][ind] = m_trial.addVar(lb=0, obj=(1-param_dic[k]['ret'][j]) * (1-param_dic[k]['g'][j])) 
    m_trial.update()   
    
    # intConstr = []
    # for k in uni_cust:
    #     intExpr = gp.quicksum(y[k][j] for j in range(len(ava_list)))
    #     intConstr.append(m_trial_constr.addConstr(intExpr == 1, name='int_'+k))
        
    # trialExpr = gp.quicksum(y[k][j] for k in uni_cust for)
    # for k in uni_cust:
        
    int_quota_list = round_to_mini(quota_list, mini)
    capacityConstr = []
    for j in ava_list:
        capacityExpr = gp.quicksum(y[k][j] for k in ind_order_df['receiver'])
        capacityConstr.append(m_trial.addConstr(capacityExpr <=  int_quota_list[j], name='capacity_%d' % j))
        
    demandConstr = []
    for (k, d) in zip(ind_order_df['receiver'], ind_order_df['weight']):
        demandExpr = gp.quicksum(y[k][j] for j in ava_list)
        demandConstr.append(m_trial.addConstr(demandExpr == d, name='demand_'+ k))
    
    m_trial.update()
    m_trial.optimize()
    
    final_all = {k:{} for k in ind_order_df['receiver']}
    for k, item in y.items():
        for j, var in item.items():
            if var.X > 0:
                final_all[k][j] = var.X
                
    return final_all


def extract_variable(dic):
    new_dic = {}
    for key, item in dic.items():
        new_dic[key] = item.X
    return new_dic


def exploit_afterparty(ind_p_dic, ind_beta_dic, cur_var_mat, u_bar, inv_bud_df, quota_by_c, cur_cust_df, tmp_remove, mini):
    rank = np.argsort(-u_bar[:,tmp_remove-1])
    w_df = cur_cust_df[cur_cust_df['group'] == tmp_remove].groupby('receiver').agg({'receiver': lambda x:x.iloc[0],
                                                                                    'weight': 'sum'})
    used_inv = np.zeros(inv_bud_df.shape[0])
    for item in quota_by_c.values():
        used_inv = used_inv + round_to_mini(item, mini)
    tmp_inv_vec = inv_bud_df['budget'].values - used_inv
    tmp_inv_vec = round_to_mini(tmp_inv_vec, mini)
    
    total_supply = 0
    total_demand = w_df['weight'].sum()
    rank_cnt = 0
    while total_supply < total_demand:
        total_supply = total_supply + tmp_inv_vec[rank[rank_cnt]]
        rank_cnt = rank_cnt + 1
        
    ava_list = rank[:rank_cnt]
    uni_cust = w_df['receiver']
    
    param_dic = {k:{} for k in uni_cust}
    for ind, k in enumerate(uni_cust):
        # mu and sigma norm
        mu = ind_beta_dic[k]['coef']['mu']
        sigma = ind_beta_dic[k]['coef']['sigma']
        
        sigma_norm = []
        for loc in ava_list:
            try:
                sigma_dec = np.linalg.cholesky(sigma)
            except np.linalg.LinAlgError as err:
                if 'positive definite' in str(err):
                    sigma_dec = np.linalg.cholesky(sigma + 1e-1*np.identity(sigma.shape[0]))
                else:
                    raise ValueError("matrix is not p.d., fail to find a chol_decom")
            sigma_norm.append(np.linalg.norm(sigma_dec @ cur_var_mat[loc,], axis=0))
        
        mu_norm = cur_var_mat[ava_list,] @ mu
        
        # return probability
        ret_prob = 1/(1+np.exp(ind_p_dic[k]['param']['u']))
        
        # g value w.r.t each possible stock
        g = [update_z(mu_norm[loc], sigma_norm[loc], 'v') + \
             update_z(-1*mu_norm[loc], sigma_norm[loc], 'v') for loc in range(len(ava_list))]
        
        param_dic[k] = {'ret': ret_prob[ava_list], 'g': g, 'admis': ava_list}
        
    m_trial = gp.Model("Afterparty Trial Assignment Constrained")
    m_trial.ModelSense = -1
    
    y = {k:{} for k in uni_cust}
    for k in uni_cust:
        for j, ind in enumerate(param_dic[k]['admis']):
            # y[k][ind] = m_trial.addVar(vtype=gp.GRB.BINARY, obj=(1-param_dic[k]['ret'][j])*param_dic[k]['g'][j])
            y[k][ind] = m_trial.addVar(lb=0, obj=(1-param_dic[k]['ret'][j]) * (1-param_dic[k]['g'][j])) 
    m_trial.update()   
        
    capacityConstr = []
    for j in ava_list:
        capacityExpr = gp.quicksum(y[k][j] for k in w_df['receiver'])
        capacityConstr.append(m_trial.addConstr(capacityExpr <=  tmp_inv_vec[j], name='capacity_%d' % j))
        
    demandConstr = []
    for (k, d) in zip(w_df['receiver'], w_df['weight']):
        demandExpr = gp.quicksum(y[k][j] for j in ava_list)
        demandConstr.append(m_trial.addConstr(demandExpr == d, name='demand_'+ k))
    
    m_trial.update()
    m_trial.optimize()
    
    final_all = {k:{} for k in w_df['receiver']}
    for k, item in y.items():
        for j, var in item.items():
            if var.X > 0:
                final_all[k][j] = var.X
    
    return final_all


def explore_afterparty(ind_p_dic, u_bar, inv_bud_df, quota_by_c, cur_cust_df, tmp_remove, mini):
    rank = np.argsort(-u_bar[:,tmp_remove-1])
    w_df = cur_cust_df[cur_cust_df['group'] == tmp_remove].groupby('receiver').agg({'receiver': lambda x:x.iloc[0],
                                                                                    'weight': 'sum'})
    used_inv = np.zeros(inv_bud_df.shape[0])
    for item in quota_by_c.values():
        used_inv = used_inv + round_to_mini(item, mini)
    tmp_inv_vec = inv_bud_df['budget'].values - used_inv
    tmp_inv_vec = round_to_mini(tmp_inv_vec, mini)
    
    total_supply = 0
    total_demand = w_df['weight'].sum()
    rank_cnt = 0
    while total_supply < total_demand:
        total_supply = total_supply + tmp_inv_vec[rank[rank_cnt]]
        rank_cnt = rank_cnt + 1
        
    ava_list = rank[:rank_cnt]
    uni_cust = w_df['receiver']
    ret_prob = {}
    for k in uni_cust:
        ret_prob[k] = 1/(1+np.exp(ind_p_dic[k]['param']['u']))
        
    m_all = gp.Model("Explore Afterparty Allocation")
    m_all.ModelSense = 1
    
    y = {k:{} for k in uni_cust}
    for k in uni_cust:
        for j in ava_list:
            y[k][j] = m_all.addVar(lb = 0, obj = ret_prob[k][j])

    m_all.update()   
    
    capacityConstr = []
    for j in ava_list:
        capacityExpr = gp.quicksum(y[k][j] for k in uni_cust)
        capacityConstr.append(m_all.addConstr(capacityExpr <=  tmp_inv_vec[j], name='capacity_%d' % j))

    demandConstr = []
    for (k, d) in zip(w_df['receiver'], w_df['weight']):
        demandExpr = gp.quicksum(y[k][j] for j in ava_list)
        demandConstr.append(m_all.addConstr(demandExpr == d, name='demand_'+ k))
    
    m_all.update()
    m_all.optimize()
    
    final_all = {k:{} for k in uni_cust}
    for k, item in y.items():
        for j, var in item.items():
            if var.X > 0:
                final_all[k][j] = var.X
    
    return final_all


def update_inventory(supply_record, tmp_inv_df, S_all):
    total_consume = np.zeros(len(S_all))
    for item in supply_record.values():
        for cust_item in item.values():
            for no, weight in cust_item.items():
                total_consume[no] += weight
    
    loc = np.concatenate([np.where(tmp_inv_df['LOTNO'] == lot)[0] for lot in S_all])
    cur_w_loc = np.where(tmp_inv_df.columns == 'budget')[0]
    tmp_inv_df.iloc[loc, cur_w_loc] = tmp_inv_df.iloc[loc, cur_w_loc].values.flatten() - total_consume
    
    return tmp_inv_df[tmp_inv_df['budget'] > 0]
        
           
def update_reaction(cur_var_mat, supply_record, offline_beta, seed):
    np.random.seed(seed)
    
    react_dic = {}
    react_cat = [-1, 1]
    
    for group_item in supply_record.values():
        for cust, item in group_item.items():
            inv_loc = []
            react = []
            q_list = []
            for loc, q in item.items():
                inv_loc.append(loc)
                p = 1/(1+np.exp(np.dot(cur_var_mat[loc,:], offline_beta[cust]['mu'])))
                react.append(np.random.choice(react_cat, size=1, p=[p,1-p]))
                q_list.append(q)
            react_dic[cust] = {'index': np.array(inv_loc), 'react': np.concatenate(react),
                               'quantity': np.array(q_list)}
    
    return react_dic


def update_ind_beta(react_dic, cur_var_mat, ind_beta_dic):
    ind_beta = copy.deepcopy(ind_beta_dic)
    for cust, item in react_dic.items():
        ret = item['index'][item['react'] == -1]
        acc = item['index'][item['react'] == 1]
        acc_logic = item['react'] == 1
        coef = ind_beta[cust]['coef']
        # weighted average
        if len(acc) > 0:
            if item['quantity'][acc_logic].sum() > 0:
                spec_w = np.dot(item['quantity'][acc_logic] / item['quantity'][acc_logic].sum(),
                                cur_var_mat[acc,:])
                try:
                    coef['mu'], coef['sigma'] = beta_update(spec_w, coef['mu'], coef['sigma'], 1)
                except np.linalg.LinAlgError as err:
                    if 'positive definite' in str(err):
                        try:
                            coef['mu'], coef['sigma'] = beta_update(spec_w, coef['mu'], 
                                                1e-5 * np.identity(len(coef['mu'])) + coef['sigma'], 1)
                        except:
                            print(cust,' fail')
                            continue
                    else:
                        raise ValueError(str(err))
            
        if len(ret) > 0:
            for k, loc in enumerate(ret):
                if item['quantity'][k].sum() > 0:
                    try:
                        coef['mu'], coef['sigma'] = beta_update(cur_var_mat[loc,:], coef['mu'],
                                                                coef['sigma'], -1)
                    except np.linalg.LinAlgError as err:
                        if 'positive definite' in str(err):
                            try:
                                coef['mu'], coef['sigma'] = beta_update(spec_w, coef['mu'], 
                                                    1e-5 * np.identity(len(coef['mu'])) + coef['sigma'], -1)
                            except:
                                print(cust,' fail')
                                continue
                        else:
                            raise ValueError(str(err))
        
        ind_beta[cust]['coef'] = coef 
        
    return ind_beta

def return_conclude(react_dic):
    total = 0
    for item in react_dic.values():
        ind = np.where(item['react']==-1)[0]
        if len(ind) > 0:
            for i in ind:
                total = item['quantity'][i] + total
    
    return total


def jsonify_beta(ind_beta_dic):
    tmp_dic = {}
    for cust, item in ind_beta_dic.items():
        mu = item['coef']['mu'].tolist()
        sigma = item['coef']['sigma'].tolist()
        tmp_dic[cust] = {'coef':{'mu':mu, 'sigma':sigma},
                         'group': str(item['group'])}
        
    return tmp_dic

def dejsonify_beta(data_beta_dic):
    beta_dic = {}
    for cust, item in data_beta_dic.items():
        mu = np.array(item['coef']['mu'])
        sigma = np.array(item['coef']['sigma'])
        group = int(item['group'])
        beta_dic[cust] = {'coef':{'mu':mu, 'sigma': sigma}, 'group': group}
        
    return beta_dic

def dejsonify_group_beta(data_beta_dic):
    beta_dic = {}
    for c, item in data_beta_dic.items():
        mu = np.array(item['coef']['mu'])
        sigma = np.array(item['coef']['sigma'])
        beta_dic[int(c)] = {'coef':{'mu': mu, 'sigma': sigma}, 'group': int(c)}
        
    return beta_dic
        
def jsonify_supply(supply_record, group_cnt = 10):
    tmp_dic = {}
    if len(supply_record) <= group_cnt: 
        tmp_dic = {int(key):{} for key in supply_record.keys()}
        for group, group_item in supply_record.items():
            for cust, item in group_item.items():
                dic = {}
                for key, val in item.items():
                    dic[int(key)] = int(val)
                tmp_dic[int(group)][cust] = dic
    else:
        for cust, item in supply_record.items():
            dic = {}
            for key, val in item.items():
                dic[int(key)] = int(val)
            tmp_dic[cust] = dic

    return tmp_dic 

def dejsonify_supply(data_supply):
    tmp_dic = {}
    for cust, item in data_supply.items():
        dic = {}
        for no, w in item.items():
            dic[int(no)] = w
        tmp_dic[cust] = dic
    
    return tmp_dic

def jsonify_inv(inv_bud_df):
    tmp_df = inv_bud_df.copy()
    tmp_df['produce date'] = [d.strftime('%Y%m%d') for d in tmp_df['produce date']]

    return tmp_df.to_dict('record') 

def dejsonify_inv(inv_dic):
    tmp_df = pd.DataFrame(inv_dic)
    if tmp_df.shape[0] > 0:
        tmp_df['produce date'] = [datetime.strptime(d, "%Y%m%d").date() for d in tmp_df['produce date']]
    
    return tmp_df
    
def jsonify_react(react_dic):
    tmp_dic = {}
    for cust, item in react_dic.items():
        tmp_dic[cust] = {'index': item['index'].tolist(),
                         'react': item['react'].tolist(),
                         'quantity': item['quantity'].tolist()}
        
    return tmp_dic

def dejsonify_react(data_react):
    tmp_dic = {}
    for cust, item in data_react.items():
        tmp_dic[cust] = {'index': np.array(item['index']),
                         'react': np.array(item['react']),
                         'quantity': np.array(item['quantity'])}

    return tmp_dic    

def jsonify_quota(quota_by_c):
    tmp_dic = {}
    for group, item in quota_by_c.items():
        tmp_dic[str(group)] = item.tolist()
        
    return tmp_dic



