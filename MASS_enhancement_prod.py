#!/usr/bin/env python
# coding: utf-8

# In[1]:


import io
import base64
import numpy as np
import pandas as pd
import ipywidgets as ipw
import IPython.display as ipd


# In[2]:


def _scorecard (score):
    if -2.00 <= score < -1.25: score = 0
    elif -1.25 <= score < -0.75: score = 1
    elif -0.75 <= score < -0.25: score = 2
    elif -0.25 <= score <= 0.25: score = 3
    elif 0.25 < score <= 0.75: score = 4
    elif 0.75 < score <= 1.25: score = 5
    elif 1.25 < score <= 2.00: score = 6    
    else: score = 7
    switcher = {
        0: [-3, 'UW_Max'],
        1: [-2, 'UW'],
        2: [-1, 'UW_Slight'],
        3: [0, 'Neutral'],
        4: [1, 'OW_Slight'],
        5: [2,'OW'],
        6: [3, 'OW_Max']}    
    return switcher.get(score, "Invalid")

def _tier_grp (score):
    if score == 1: score = 0
    elif score == 2: score = 1
    elif score == 3: score = 2
    elif score == 4: score = 3
    elif score == 5: score = 4
    switcher = {
        0:'Tier_A',
        1:'Tier_B',
        2:'Tier_C',
        3:'Tier_D',
        4:'Tier_E'}    
    return switcher.get(score, "Invalid")


# In[3]:


def _un_wgt(rnd):

    feature = rnd.values[:-1]
    saa = feature[:,0]
    score = feature[:,1]
    urng = feature[:,2]
    lrng = feature[:,3]
    prio = feature[:,4]

    tilt = []
    for i in score:
        tilt.append(_scorecard(i)[0])
    
    counter = 0
    app_tilt = np.array([])
    for i in tilt:
        if i > 0:
            t = saa[counter] + ((urng[counter]-saa[counter])/3*abs(i))
            app_tilt = np.append(app_tilt,t)
        elif i < 0:
            t = saa[counter] - ((saa[counter]-lrng[counter])/3*abs(i))
            app_tilt = np.append(app_tilt,t)
        else:
            app_tilt = np.append(app_tilt,saa[counter])
        
        counter += 1
        
    feature = np.append(feature,app_tilt.reshape(-1,1),axis=1)
    feature = np.append(feature,app_tilt.reshape(-1,1),axis=1)
    
    return feature


# In[84]:


def _available_gap(feature,rnd):

    un_wgt = feature[:,5]
    cash = rnd.values[-1]
    cash_saa = cash[0]
    cash_urng = cash[2]
    cash_lrng = cash[3]
    
    saa = feature[:,0]
    score = feature[:,1]
    urng = feature[:,2]
    lrng = feature[:,3]
    prio = feature[:,4]
    
    wgt_bal = {}
    gap = {}
    tier = 5
    for i in range(1,tier+1):
        wgt_bal[i] = 0.00
        gap[i] = 0.00

    def avail_gap():
        gap = 0
        if un_wgt.sum() + cash_urng < 1:
            gap += urng[counter] - un_wgt[counter]
        elif un_wgt.sum() + cash_lrng > 1:
            gap += un_wgt[counter] - lrng[counter]
        else: gap = 0
        return gap
    
    counter = 0
    for i in prio:
        if i == 1:
            wgt_bal[1]+=un_wgt[counter]
            gap[1] = 0
        elif i == 2:
            wgt_bal[2]+=un_wgt[counter]
            gap[2] += avail_gap()
        elif i == 3:
            wgt_bal[3]+=un_wgt[counter]
            gap[3] += avail_gap()
        elif i == 4:
            wgt_bal[4]+=un_wgt[counter]
            gap[4] += avail_gap()
        elif i == 5:
            wgt_bal[5]+=un_wgt[counter]
            gap[5] += avail_gap()
        counter+=1
        
    bucket = np.array(list(wgt_bal.values()))
    gap_bucket = np.array(list(gap.values()))
    
    return bucket, gap_bucket


# In[5]:


def asset_dict(rnd,feature):
    sec_level_dict = {}
    counter = 0
    for i in rnd.index[:-1]:
        sec_level_dict[i] = feature[counter]
        counter += 1
    return sec_level_dict

def tier_dict(dct,num_range):
    def sort_dict(i,dct):
        tier_dict = {}
        for k,v in dct.items():
            if v[4] == i:
                tier_dict[k] = v
        return tier_dict
    
    order_dict = {}
    order = list(range(1,num_range+1))[::-1]
    for i in order:
        order_dict[i] = sort_dict(i,dct)    
    
    return order_dict


# In[83]:


def sec_level_wgt(rnd,dct,num_range,bucket,gap_bucket): 
    
    order = list(range(1,num_range+1))[::-1]
    
    cash = rnd.values[-1]
    cash_saa = cash[0]
    cash_urng = cash[2]
    cash_lrng = cash[3]
    
    balance = 1-bucket.sum()
    
    
    def _excess(balance):
        
        dct_copy = dct.copy()
        
        def _excess_rebalancer(dct,tier,_bal):
            
            gb_index = tier-1 
            check_gap = 0
            sec_roll_bal = 0
            
            for k,v in dct[tier].items():
                if gap_bucket[gb_index]>_bal:
                    check_gap = _bal
                else:
                    check_gap = gap_bucket[gb_index] 
                
                base_alloc = dct_copy[tier][k][-1]
                
                sec_level_gap = v[5] - v[3]
                sec_level_increment = (check_gap * (base_alloc/bucket[gb_index])) #(sec_level_gap/gap_bucket[gb_index]))
                np.seterr('ignore')
                sec_level_alloc = base_alloc - sec_level_increment
                
                #print(base_alloc,check_gap,sec_level_gap,gap_bucket[gb_index])
                
                v[-1] = sec_level_alloc
        
                if np.isnan(sec_level_alloc) or np.isinf(sec_level_alloc):
                    v[-1] = base_alloc
                else:
                    v[-1] = sec_level_alloc
                
                sec_roll_bal += sec_level_increment
                
            return sec_roll_bal
        
        roll_bal = balance
        for i in order:
                
            per_tier_balance = _excess_rebalancer(dct,i,roll_bal)
            
            if np.isnan(per_tier_balance):
                per_tier_balance = 0
                roll_bal -= per_tier_balance
            else: 
                roll_bal -= per_tier_balance
            
        return dct
    
    
    def _residual(balance):
        
        dct_copy = dct.copy()
        
        def _residual_rebalancer(dct,tier,_bal):
        
            gb_index = tier-1 
            check_gap = 0
            sec_roll_bal = 0
    
            for k,v in dct[tier].items():
                if gap_bucket[gb_index]>_bal:
                    check_gap = _bal
                else:
                    check_gap = gap_bucket[gb_index]
                    
                base_alloc = dct_copy[tier][k][-1]
    
                sec_level_gap = v[2] - v[5]
                sec_level_increment = (check_gap * (base_alloc/bucket[gb_index])) #(sec_level_gap/gap_bucket[gb_index]))
                np.seterr('ignore')
                sec_level_alloc = base_alloc + sec_level_increment
                
                v[-1] = sec_level_alloc
                
                if np.isnan(sec_level_alloc) or np.isinf(sec_level_alloc):
                    v[-1] = base_alloc
                else:
                    v[-1] = sec_level_alloc
                
                sec_roll_bal += sec_level_increment
                
            return sec_roll_bal
    
        roll_bal = balance
        for i in order:
            per_tier_balance = _residual_rebalancer(dct,i,roll_bal)
            
            if np.isnan(per_tier_balance):
                per_tier_balance = 0
                roll_bal -= per_tier_balance
            else: 
                roll_bal -= per_tier_balance
            
        return dct
    
    _res = 1 - (bucket.sum() + cash_urng)
    _exc = (bucket.sum() + cash_lrng) - 1
    
    wgt_low = 1 - cash_urng
    wgt_high = 1 - cash_lrng
    
    if wgt_low <= bucket.sum() <= wgt_high:
        final_cash = 1 - bucket.sum()
        
        print('Final weight is within range, No reallocation required')
        print('Cash balance: ', (final_cash*100).round(4), ' Within Cash Min-Max Limits' )
        
        return dct
        
    
    elif 0 < _res < 1:
        balance = 1 - (bucket.sum() + cash_urng)
        print('Residual Balance: ',(balance*100).round(4))
        
        return _residual(balance)
    
    elif _exc > 0:
        balance = (bucket.sum() + cash_lrng) - 1
        print('Excess Balance: ',(balance*100).round(4))
        
        return _excess(balance)
    
    elif _res == 0 and _exc == 0:
        print('Initial allocation within limits')
        return dct
    
    else:
        print('recheck excess/residual balances')
        print('resid: ', 1 - (bucket.sum() + cash_urng))
        print('excess: ', (bucket.sum() + cash_lrng) - 1)
        
        return dct


# In[60]:


def _post_upload_check(src_df):
    
    df = src_df.iloc[:-1,:]
    cash = src_df.iloc[-1,:]
    
    assert src_df.columns[0] == 'port_alloc', 'Column heading should be port_alloc'
    assert src_df.columns[1] == 'class_score', 'Column heading should be class_score'
    assert src_df.columns[2] == 'up_rng', 'Column heading should be up_rng'
    assert src_df.columns[3] == 'low_rng', 'Column heading should be low_rng'
    assert src_df.columns[4] == 'port_prio', 'Column heading should be port_prio'
    assert src_df.columns[5] == 'sector', 'Column heading should be sector'
    
    if src_df['port_alloc'].sum().round(2) != 1.00:
        print('Input Data should be in decimal form')
        raise Exception('Input Data should be in decimal form')
    
    if len(src_df.columns) > 6:
        print('Input Data columns > 5, delete extra columns')
        raise Exception('Input Data columns > 5, delete extra columns')
        
    for index, row in src_df.iterrows():
        if np.isnan(row['port_alloc']):
            print('Blank Row detected')
            raise Exception('Blank Row detected')            
    
    for index, row in df.iterrows():
        if row['class_score'] < -2.00 or row['class_score'] > 2.00:
            print(row.name, ' Score should be within -2 and +2')
            raise Exception('Score should be within -2 and +2')
    
    for index, row in df.iterrows():
        if row['port_prio'] not in [1,2,3,4,5]:
            print(row.name, ' Priority Values should be integer [1,2,3,4,5]')
            raise Exception('Priority Values should be integer [1,2,3,4,5]')
    
    if cash.name not in ['Cash','cash']:
        print('Add Cash Values')
        raise Exception('Add Cash Values')
    else:
        if not np.isnan(cash['class_score']) and not np.isnan(cash['port_prio']):
            print('Cash should not have score and priority')
            raise Exception('Cash should not have score and priority')
            
    if np.isnan(cash['port_alloc']) or np.isnan(cash['up_rng']) or np.isnan(cash['low_rng']):
        print('Cash Values should at least be Zero')
        raise Exception('Cash Values should at least be Zero')
            
    return None


def get_inputs(x_dict):
    inputs_dict = {}
    for key,widget in x_dict.items():
        if hasattr(widget,'value'):
            inputs_dict[key] = widget.value
    return inputs_dict


def df_clean(weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict):
    
    input_data = pd.DataFrame([get_inputs(weight_dict).values(),
                               get_inputs(score_dict).values(),
                               get_inputs(u_rng_dict).values(),
                               get_inputs(l_rng_dict).values(),
                               get_inputs(prio_dict).values()],
                              columns=get_inputs(weight_dict).keys()).transpose()
    input_data.rename(columns={0:'port_alloc',1:'class_score',2:'up_rng',3:'low_rng',4:'port_prio',},inplace=True)
    
    input_data['port_alloc'] = input_data['port_alloc']/100
    input_data['up_rng'] = input_data['up_rng']/100
    input_data['low_rng'] = input_data['low_rng']/100
    
    return input_data




def _input_data_checker(df):
    
    cash_net = df.iloc[:-1,:]
    cash_spec = df.values[-1]
    max_cash = cash_spec[2]
    min_taa = cash_net['low_rng'].sum()
    
    if df['port_alloc'].sum().round(2) != 1:
        print('Check SAA Total: ',df['port_alloc'].sum().round(2))
        raise Exception('Strategic Asset Allocation must equal 100')
    
    elif df['port_alloc'].sum().round(2) > 1:
        print('Total Strategic Asset Allocation > 100%')
        raise Exception('Total Strategic Asset Allocation > 100%')
    
    elif df['port_alloc'].sum() == 0:
        print('No Input Data Detected')
        raise Exception('No Input Data Detected')

    elif max_cash > (1-min_taa).round(4): 
        print('Adjust Cash Levels - Max Cash exposure > (100 - Total of Minimum TAAs)')
        raise Exception('Error: Adjust Cash Maximum TAA')
    
    for index, row in df.iterrows():
        if row['up_rng'] < row['port_alloc'] or row['low_rng'] > row['port_alloc']:
            print(row.name, ' Check Max and Min')
            raise Exception('Check Max and Min')
            
    else: print('Input data conditions met')
    
    return None
            

def _balance_check():
    port_sum = 0
    for k,v in final_order_dict.items():
        for x,y in v.items():
            port_sum += y[-1]
            print(x,y[4],y[-1].round(4)*100)
    return port_sum.round(4)

def _formatter(dct):
    asset_list = []
    value_list = []
    for k,v in dct.items():
        for x,y in v.items():
            asset_list.append(x)
            value_list.append(y)
    df = pd.DataFrame(value_list,index=asset_list,
                      columns=['saa','score','urng','lrng',
                               'prio','init_wgt','final_wgt']).sort_values(by='prio',ascending=True)
    return df


def _cat_tier_check(df):
    category_list =['UW_Max','UW','UW_Slight','Neutral','OW_Slight','OW','OW_Max']
    tier_list = ['Tier_A','Tier_B','Tier_C','Tier_D','Tier_E']
    
    _cat = []
    _tier = []
    for index, row in df.iterrows():
        _cat.append(_scorecard(row['score'])[1])
        _tier.append(_tier_grp(row['prio']))
        
    df['_cat'] = _cat
    df['_tier'] = _tier
    
    for cat in category_list:    
        if df['_cat'].eq(cat).all():
            for tier in tier_list:
                if df['_tier'].eq(tier).all(): 
                    df['final_wgt'] = df['saa'] 
                    print('All categories and tiers are equal, revert to SAA')
    return df


def _tierA_check(df):
    if df[df['prio'] == 1]['init_wgt'].sum() > 1:
        raise Exception("Error: Re-classify Tier A, Final holdings of Tier A assets exceeds 100%")
    return df



def _final_assertion(dct,cash,cash_lrng,cash_urng):
    
    checks = _formatter(dct).sum()

    chk_fwgt = checks[-1].round(2)
    chk_ex = (1 - cash_lrng).round(2)
    chk_res = (chk_fwgt + cash_urng).round(2)

    if chk_fwgt > chk_ex:
        print("Warning: Excess holdings needs to be reduced, re-adjust tiers, cash, and/or TAA low range")
    elif chk_res < 1:
        print("Warning: Residual holdings needs to be bought, re-adjust tiers, cash, and/or TAA upper range")
    
    else: print('All rebalance conditions met')

    print('Initial Total Unrestrained Weight (Ex Cash): ',(checks[-2]*100).round(4))
    print('Total Final Weight (Ex Cash): ',(checks[-1]*100).round(4))
    print('Cash SAA: ',(cash[0]*100).round(4))
    print('Cash Max: ',(cash[2]*100).round(4))
    print('Cash Min: ',(cash[3]*100).round(4))
    
    #print(checks)
    
    return None


def _viz_output(df):
    
    df['saa'] = df['saa'] * 100
    df['urng'] = df['urng'] * 100
    df['lrng'] = df['lrng'] * 100
    df['init_wgt'] = df['init_wgt'] * 100
    df['final_wgt'] = df['final_wgt'] * 100
    df['delta'] = df['final_wgt'] - df['init_wgt']
    
    df.rename(columns={'saa':'Strategic Asset %',
                        'urng':'Max Tactical %',
                        'lrng':'Min Tactical %',
                        '_cat':'Score Category',
                        '_tier':'Tier Bucket',
                        'init_wgt':'Unrestrained %',
                        'final_wgt':'Final Asset %',},inplace=True)
    
    format_titles = ['Strategic Asset %','Min Tactical %','Max Tactical %','Score Category',
                     'Tier Bucket','Unrestrained %','Final Asset %','delta']
    
    viz_df = df.reindex(columns=format_titles)
    
    return viz_df.round(2)


# In[61]:


def _MASS_algo(rnd,upload_file):
    
    sec_df = upload_file[:-1]['sector']
    
    _input_data_checker(rnd)
    
    cash = rnd.values[-1]
    cash_saa = cash[0]
    cash_urng = cash[2]
    cash_lrng = cash[3]

    feature = _un_wgt(rnd)
    bucket, gap_bucket = _available_gap(feature,rnd)
    order_dict = tier_dict(asset_dict(rnd,feature),5)
    final_order_dict = sec_level_wgt(rnd,order_dict,5,bucket, gap_bucket)
    clean_output = _tierA_check(_cat_tier_check(_formatter(final_order_dict)))
    
    viz_df = _viz_output(clean_output)
    viz_df['sector'] = sec_df
    viz_df.sort_values(['sector','Tier Bucket'],inplace=True)
    
    final_cash = (100 - viz_df['Final Asset %'].sum()).round(2)
    
    cash_append = np.array([cash_saa*100,' ',cash_urng*100,cash_lrng*100,' ','',final_cash,' ','Cash'])
    cash_df = pd.DataFrame(cash_append,columns=['CASH'],index = ['Strategic Asset %',
                                                          'Score Category',
                                                          'Max Tactical %',
                                                          'Min Tactical %',
                                                          'Tier Bucket',
                                                          'Unrestrained %',
                                                          'Final Asset %',
                                                          'delta',
                                                          'sector']).transpose()
    viz_df = pd.concat([viz_df,cash_df],axis=0)
    
    _final_assertion(final_order_dict,cash,cash_lrng,cash_urng)
    
    return viz_df.round(2)


# In[9]:


def _init_UI (src):
        
    upload_file = src.set_index(src.columns[0])
    
    _post_upload_check(upload_file)
    
    sector_list = list(upload_file['sector'].drop_duplicates().dropna())
        
    layout = ipw.Layout(width ='150px')
    layout_p = ipw.Layout(width ='175px')
    options = [('Tier A',1),('Tier B',2),('Tier C',3),('Tier D',4),('Tier E',5)]
    value = 2
    score_min = -2
    score_max = 2
    step = 0.01
    
    
    heading = ipw.HTML('''<b><h18> Security Level Asset Scoring System </h18></b>''')
    saa = ipw.HTML('''<b><h18> SAA Weight </h18></b>''',layout=layout)
    tilt = ipw.HTML('''<b><h18> Tilt Score </h18></b>''',layout=layout)
    max_taa = ipw.HTML('''<b><h18> Max TAA </h18></b>''',layout=layout)
    min_taa = ipw.HTML('''<b><h18> Min TAA </h18></b>''',layout=layout)
    tier = ipw.HTML('''<b><h18> Tier Category </h18></b>''',layout=layout)
    
    title = ipw.HBox([heading])
    space = ipw.HBox([],layout=ipw.Layout(height='20px'))
    
    calc = ipw.Button(description="Calculate")
    clear = ipw.Button(description="Clear")
    refresh = ipw.Button(description="Refresh")
    extract = ipw.Button(description="Extract")
    
    output = ipw.Output()

    def calc_button(b):
        with output:
            output.clear_output()
            display(_MASS_algo(df_clean(weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict),upload_file))
            
    def clear_button(b):
        with output:
            for i in _saa_list: i.value = 0
            for i in _min_taa: i.value = 0
            for i in _max_taa: i.value = 0
            for i in _score: i.value = 0
            for i in _prio: i.value = 2
            CLASS_CASH.value = 0
            uCLASS_CASH.value = 0
            lCLASS_CASH.value = 0
            output.clear_output()    
    
    def refresh_button(b):
        with output:
            counter = 0 
            for i in _saa_list: 
                i.value = sector_sort['port_alloc'][counter]*100
                counter += 1
    
            counter = 0 
            for i in _min_taa:
                i.value = sector_sort['low_rng'][counter]*100
                counter += 1
    
            counter = 0 
            for i in _max_taa:
                i.value = sector_sort['up_rng'][counter]*100
                counter += 1

            counter = 0 
            for i in _score:
                i.value = sector_sort['class_score'][counter]
                counter += 1

            counter = 0 
            for i in _prio:
                i.value = sector_sort['port_prio'][counter]
                counter += 1
        
            CLASS_CASH.value = upload_file.iloc[-1][0] * 100
            uCLASS_CASH.value = upload_file.iloc[-1][2] * 100
            lCLASS_CASH.value = upload_file.iloc[-1][3] * 100

    
    def extract_button(b):
        with output:
            output.clear_output()
            
            rnd = _MASS_algo(df_clean(weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict),upload_file)
            df = rnd.reset_index()
            
            buffer = io.BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)

            b64 = base64.b64encode(buffer.read()).decode()
            href = f'<a href="data:application/csv;base64,{b64}" download="rebal_file.csv">Download</a>'

            display(ipw.HTML(href))
    
    
    calc.on_click(calc_button)
    clear.on_click(clear_button)
    refresh.on_click(refresh_button)
    extract.on_click(extract_button)
    
    
    CLASS_CASH = ipw.BoundedFloatText(min=0, style={'description_width':'initial'},layout=layout,description='Cash SAA')
    uCLASS_CASH = ipw.BoundedFloatText(min=0, style={'description_width':'initial'},layout=layout,description='MAX Cash')
    lCLASS_CASH = ipw.BoundedFloatText(min=0, style={'description_width':'initial'},layout=layout,description='MIN Cash')
    
    conso_asset_container = ipw.VBox([])

    weight_dict = {}
    l_rng_dict = {}
    u_rng_dict = {}
    score_dict = {}
    prio_dict = {}

    _saa_list = []
    _min_taa = []
    _max_taa = []
    _score = []
    _prio = []

    
    def sub_asset_list(sub_list):
        asset_grp = ipw.HBox([])
        asset_container = ipw.VBox([])
        
        for n in sub_list:
            asset_saa = ipw.BoundedFloatText(min=0,description=n,layout=layout)
            minimum = ipw.BoundedFloatText(min=0,description='Min',layout=layout)
            maximum = ipw.BoundedFloatText(min=0,description='Max',layout=layout)
            score = ipw.BoundedFloatText(min=score_min,max=score_max,step=step,description='Score',layout=layout)
            prio = ipw.Dropdown(options=options,value=value,description=n,layout=layout_p)

            _saa_list.append(asset_saa)
            _min_taa.append(minimum)
            _max_taa.append(maximum)
            _score.append(score)
            _prio.append(prio)

            asset_grp = ipw.HBox([asset_saa,minimum,maximum,score,prio])
            asset_container = ipw.VBox([asset_container, asset_grp])
            
        return asset_container
            
    
    sector_sort = pd.DataFrame()
    sec_df = upload_file[:-1]
    
    for n in sector_list:
        sect = ipw.HTML(n,layout=layout)
        sub_title = ipw.HBox([sect])
        sub_df = sec_df[sec_df['sector']==n]
        sub_list = list(sub_df.index)
        
        asset_container = sub_asset_list(sub_list)
        
        title_asset_container = ipw.VBox([space,sub_title,asset_container])
        conso_asset_container = ipw.VBox([conso_asset_container,title_asset_container])
        
        sector_sort = pd.concat([sector_sort,sub_df])
    
    
    counter = 0 
    for i in _saa_list: 
        i.value = sector_sort['port_alloc'][counter]*100
        counter += 1

    counter = 0 
    for i in _min_taa:
        i.value = sector_sort['low_rng'][counter]*100
        counter += 1

    counter = 0 
    for i in _max_taa:
        i.value = sector_sort['up_rng'][counter]*100
        counter += 1

    counter = 0 
    for i in _score:
        i.value = sector_sort['class_score'][counter]
        counter += 1

    counter = 0 
    for i in _prio:
        i.value = sector_sort['port_prio'][counter]
        counter += 1

    CLASS_CASH.value = upload_file.iloc[-1][0] * 100
    uCLASS_CASH.value = upload_file.iloc[-1][2] * 100
    lCLASS_CASH.value = upload_file.iloc[-1][3] * 100
    
    
    asset_list = list(sector_sort.index)
    counter = 0
    for n in asset_list:
        weight_dict[n] = _saa_list[counter]
        l_rng_dict[n] = _min_taa[counter]
        u_rng_dict[n] = _max_taa[counter]
        score_dict[n] = _score[counter]
        prio_dict[n] = _prio[counter]
        counter += 1
    
    cash_title = ipw.HTML('CASH',layout=layout)
    cash_heading = ipw.HBox([cash_title])
    cash_cont = ipw.HBox([CLASS_CASH, space, lCLASS_CASH, uCLASS_CASH])
    conso_cash = ipw.VBox([cash_heading,cash_cont])
    
    buttons = ipw.HBox([calc,clear,refresh,extract]) 
    
    sub_head = ipw.HBox([saa,min_taa,max_taa,tilt,tier])
    
    container = ipw.VBox([title, space, sub_head, conso_asset_container, space, conso_cash, space, buttons, output])
    
    weight_dict['CASH'] = CLASS_CASH
    u_rng_dict['CASH'] = uCLASS_CASH
    l_rng_dict['CASH'] = lCLASS_CASH
    
    return container, weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict


# In[65]:


def _init_MASS():

    def file_upload_handler(change):
        upload_file = change['new']
        file_content = upload_file[list(upload_file.keys())[0]]['content']
        data_file = io.BytesIO(file_content)
        input_data = pd.read_csv(data_file)
        ui, weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict = _init_UI(input_data)
        display(ui)
    
    file_upload = ipw.FileUpload(accept=".csv", multiple=False)
    file_upload.observe(file_upload_handler, names='value')
    display(file_upload)
    
def _init_MASS_direct(data_file):
    input_data = data_file
    ui, weight_dict, score_dict, u_rng_dict, l_rng_dict, prio_dict = _init_UI(input_data)
    display(ui)
    



