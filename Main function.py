
import json
import pandas as pd
import numpy as np
import copy
# import datetime


### parameters
# import data
data_location = 'C:\\Users\\eric8\\Desktop\\graduate thesis reference\\Numerical'
os.chdir(data_location)
MAT = '757XXXX'

# duration information
start_str = '20140101'
end_str = '20181231'
initial_str = '20140630'

# model parameter
backward = 90
limit_weight = 2000
keep_spec = ['色相(調) - YI', '污點', '拉力強度 - TSy ( 50 mm/min)', '拉力強度 - TSb ( 50 mm/min)',
             '流動係數   MVR  220℃×10kg', 'Charpy Impact', '拉力強度 - EL ( 50 mm/min)']
square_spec = ['色相(調) - YI', '拉力強度 - TSy ( 50 mm/min)', '拉力強度 - TSb ( 50 mm/min)',
               '流動係數   MVR  220℃×10kg', 'Charpy Impact', '拉力強度 - EL ( 50 mm/min)']

alpha = 0.5
E1 = 0
E2 = 5
E3 = 1
thres = 1e-4

seed = 59487
standardized = True
verbose = True

### main function
# import data
spec_inform, deal_inform = import_data(data_location)
spec, deal = lot_based_spec_customer_df(spec_inform, deal_inform, MAT)
duration = duration_inform(spec)

start, initial, model_start, end, MonSun = generate_date(start_str, initial_str, end_str)

spec_df = spec[np.logical_and(spec['produce date'] >= start,
                              spec['produce date'] <= end)].dropna(axis=1)

deal_df = deal[np.logical_and(deal['produce date'] >= start,
                             deal['produce date'] <= end)].dropna(axis=1)

var_dic, std_dic = input_var_matrix(spec_df, keep_spec, square_spec, standardized)

group_initial_beta, ind_initial_beta = generate_initial_beta(initial, cust_deal_df, \
                                         return_summary, std_dic, keep_spec, square_spec)
    
global_message_dic = {}
## ind_true_beta
offline_beta = estimate_offline_beta(start, end, cust_deal_df, ind_initial_beta,
                                     std_dic, keep_spec, square_spec)

# re-run start from here
result_record = {}

## the inventory level after warmup
inv_bud_df, leftover_df = initial_current_weight_update(start, initial, cust_deal_df, spec_df, limit_weight)

# cur_spec_Df!!!!

ind_beta_dic = copy.deepcopy(ind_initial_beta)
weighting_beta_dic = copy.deepcopy(group_initial_beta)
for period in range(len(MonSun)):
    if period % 2 == 0:
        print(period)
        if len(result_record) == 0:
            print('A whole new world!')
        elif period < list(result_record.keys())[-1]:
            continue
        elif period == list(result_record.keys())[-1]:
            f = open(MonSun[period].strftime('%Y%m%d') + '.json', "r")
            data = json.loads(f.read())
            
            ind_beta_dic = dejsonify_beta(data['beta'])
            inv_bud_df = dejsonify_inv(data['inventory'])
            weighting_beta_dic = dejsonify_group_beta(data['group beta'])
            continue
        
        seed = seed + period
        start_date = MonSun[period]
        end_date = MonSun[period+1]
        global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = \
            'start to run the case: start in ' + start_date.strftime('%Y/%m/%d')
            
        for cust, item in ind_beta_dic.items():
            if np.isnan(item['coef']['mu']).sum() > 0:
                print(cust)
                raise ValueError("Fuck you NO.1")
                
        # a = update_ind_beta(react_dic, cur_var_mat, ind_beta_dic)
        # for cust, item in a.items():
        #     if np.isnan(item['coef']['mu']).sum() > 0:
        #         print(cust)
        #         # raise ValueError("Fuck you NO.1")
        
        # current inventory, customer information
        # cur_cust_df, inv_bud_df, S_all = cur_cust_spec_df(start_date, end_date, cust_deal_df, cur_spec_df)
        cur_cust_df, inv_bud_df, S_all, leftover_df = cur_cust_spec_df(start_date, end_date, cust_deal_df,
                                                                       inv_bud_df, leftover_df, spec_df, limit_weight)
        
        # update group beta
        ind_beta_dic = new_comer_beta_update(cur_cust_df, ind_beta_dic, weighting_beta_dic)
        weighting_beta_dic, total_beta_dic = update_group_beta(backward, end_date, cur_cust_df, ind_beta_dic)
        
        global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = weighting_beta_dic[7]['coef']['mu']
        
        # the BLP estimation under current inventory and beta
        ind_p_dic = BLP_prob(S_all, ind_beta_dic, var_dic)
        group_p_dic = BLP_prob(S_all, weighting_beta_dic, var_dic)
        all_p_dic = BLP_prob(S_all, total_beta_dic, var_dic)
        
        # solving for best assortment for each group
        pre_ratio = [1.05 for c in range(len(weighting_beta_dic))]
        total_quota, quota_by_c, tmp_remove, u_bar = solve(cur_cust_df, inv_bud_df, S_all, var_dic,
                                                           ind_p_dic, group_p_dic, all_p_dic, limit_weight, pre_ratio, seed,
                                                           alpha, E1, E2, E3, thres, iter_limit = 200)

        # allocation
        mini = 100
        cur_var_mat = cur_var_matrix(var_dic, inv_bud_df)
        # pre_allocation
        supply_record = {}
        for c in range(len(weighting_beta_dic)):
            ind_order_df = individual_order(cur_cust_df, c)
            supply_record[c] = pre_allocation(ind_p_dic, quota_by_c[c], ind_order_df, mini)
            # exploit_all_dic = trial_assign(ind_p_dic, ind_beta_dic, cur_var_mat, quota_list, prior_all, ind_order_df)#, delta_list)
            # y, para = trial_assign(ind_p_dic, ind_beta_dic, cur_var_mat, quota_by_c[c], [], ind_order_df, mini)
        
        supply_record = dict(ele for sub in supply_record.values() for ele in sub.items())
        # supply_record = product_allocation(ind_p_dic, ind_beta_dic, quota_by_c, cur_cust_df, cur_var_mat, tmp_remove)
        # if tmp_remove != 0:
        #     supply_record = afterparty(ind_p_dic, u_bar, inv_bud_df, supply_record, cur_cust_df, tmp_remove)
        inv_bud_df = update_inventory(supply_record, inv_bud_df, S_all)
        
        # inv_bud_df = update_inventory(supply_record, inv_bud_df, S_all)
        # aa = update_inventory(supply_record, inv_bud_df, S_all)
        
        # get the reaction
        react_dic = update_reaction(cur_var_mat, supply_record, offline_beta, seed)

        # update beta
        ind_beta_dic = update_ind_beta(react_dic, cur_var_mat, ind_beta_dic)
        # zzz = update_ind_beta(react_dic, cur_var_mat, ind_beta_dic)
        
        for cust, item in ind_beta_dic.items():
            if np.isnan(item['coef']['mu']).sum() > 0:
                print(cust)
                raise ValueError("Fuck you NO.2")
        
        # store result
        cur_record = {'supply': jsonify_supply(supply_record),
                      'inventory': jsonify_inv(inv_bud_df),
                      'reaction': jsonify_react(react_dic),
                      'beta': jsonify_beta(ind_beta_dic.copy()),
                      'group beta':jsonify_beta(weighting_beta_dic.copy())}
        
        with open(MonSun[period].strftime('%Y%m%d')+".json", "w") as write_file:
            json.dump(cur_record, write_file, indent=4)
        
        result_record[period] = {'supply': supply_record,
                                 'inventory': inv_bud_df,
                                 'reaction': react_dic,
                                 'beta': ind_beta_dic.copy(),
                                 'group beta': weighting_beta_dic.copy()}
        
        global_message_dic[datetime.now().strftime("%m/%d/%Y, %H:%M:%S")] = \
            'end of running the case: end in ' + end_date.strftime('%Y/%m/%d')
        
        
        with open("Message.txt", 'w') as f: 
            for key, value in global_message_dic.items(): 
                f.write('%s:%s\n' % (key, value))
        

 








# aaa = inv_bud_df.merge(spec_df[['LOTNO']+keep_spec], how='inner', on='LOTNO')
# aaa[['LOTNO']+keep_spec].duplicated()

tmp_dic = {(c+1):[] for c in range(10)}
for i in range(0,50,2):
    f = open(MonSun[i].strftime('%Y%m%d') + '.json', "r")
    data = json.loads(f.read())
    
    g_beta = dejsonify_group_beta(data['group beta'])
    for c in range(10):
        tmp_dic[c+1].append(g_beta[c+1]['coef']['mu'])

for c in range(10):
    tmp_dic[c+1] = np.array(tmp_dic[c+1])
    

for i in range(0,30,2):
    print(i)
    for cust, item in ttt[i]['beta'].items():
        if np.isnan(item['coef']['mu']).sum() > 0:
            print(cust, '  group:', item['group'])
            
for t in range(0,30,2):
    for cust, item in result_record[t]['beta'].items():
        if np.isnan(item['coef']['mu']).sum() > 0:
            print(cust, '  group:', item['group'])

for cust, item in ind_beta_dic.items():
    if np.isnan(item['coef']['mu']).sum() > 0:
        print(cust, '  group:', item['group'])
    
            
tmp_dic = {}
for t in range(0,30,2):
    tmp_list = []
    cust_list = []
    for cust, item in result_record[t]['beta'].items():
        if item['group'] == 7:
            tmp_list.append(item['coef']['mu'])
            cust_list.append(cust)
    tmp_mat = np.append(np.array(cust_list).reshape((len(cust_list),1)), np.array(tmp_list), axis=1)
    tmp_dic[t] = tmp_mat
    
tmp_list = []
cust_list = []
for cust, item in ind_beta_dic.items():
    if item['group'] == 7:
        tmp_list.append(item['coef']['mu'])
        cust_list.append(cust)
tmp_mat = np.append(np.array(cust_list).reshape((len(cust_list),1)), np.array(tmp_list), axis=1)

fff = pd.DataFrame(tmp_mat)


tmp_list = []
cust_list = []
r_dic = {}
for cust, item in zzz.items():
    if item['group'] == 7 and cust in react_dic.keys():
        tmp_list.append(item['coef']['mu'])
        cust_list.append(cust)
        r_dic[cust] = react_dic[cust]['index']
tmp_mat = np.append(np.array(cust_list).reshape((len(cust_list),1)), np.array(tmp_list), axis=1)

yyy = pd.DataFrame(tmp_mat)








# result_record_alpha5 = result_record
# result_record_alpha9 = copy.deepcopy(result_record)
# result_record_alpha1 = copy.deepcopy(result_record)





        
    