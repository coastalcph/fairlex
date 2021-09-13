import pandas as pd
import os

def format_results_all_data_splits(eval_folder, protected_attribute):
    df = None
    splits = ['train', 'val', 'test']
    for s in splits:
        file_path = os.path.join(eval_folder, f'{s}_eval.csv')
        d = format_results(file_path, protected_attribute)

        # d.columns = [f'{s}-{name}' for name in d.columns]
        if d is None:
            print('PROBLEMS READING THE FILE')
            return
                
        if df is None:
            df = d
        else:
            # df = df.merge(d.reset_index(drop=True), how='left', on=df.index)
            df = df.merge(d, right_index=True, left_index=True)
    df.columns =  ['train_f1', 'train_support', 'val_f1', 'val_support', 'test_f1', 'test_support']
    return df
    
    
def format_results(eval_test_file, protected_attribute, do_print=False):
    if not os.path.exists(eval_test_file):
        return None
    try:
        test_data:pd.DataFrame = pd.read_csv(eval_test_file)
    except:
        return None
    
    test_data = test_data.iloc[-1] # take best epoch
    supports = list(test_data.filter(regex='.*count.*').values)
    supports = [-1, sum(supports)] + supports
    test_data = test_data.transpose()
    columns_to_drop = [x for x in test_data.index.values if 'count' in x or 'wg' in x]
    test_data = test_data.drop(columns_to_drop)
    test_data = test_data.rename(lambda label: label.replace("F1-micro_", "").replace(protected_attribute+':', ""))
    # test_data['supports'] = supports
    test_data = test_data.to_frame().assign(supports=pd.Series(supports).values)
    labels = list(test_data.index.values)
    all_idx = labels.index('all')
    del labels[all_idx]
    labels.append('all')
    test_data = test_data.reindex(labels)
    test_data.columns = ['micro-f1', 'supports']
    if do_print:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):  # more options can be specified also
            print(test_data.to_string(index=False))
    return test_data


if __name__ == '__main__':
    algorithms = ['groupDRO', 'ERM', 'deepCORAL']
    splits = ['official', 'temporal', 'uniform']
    protected_groups = ['issue_area']

    for pg in protected_groups:
        for s in splits:
            for a in algorithms:
                camel_case_pg = pg.split('_')[0] + pg.split('_')[1].capitalize()
                print(f"{pg.upper()}\t{a.upper()}\t{s.upper()}")
                df = format_results_all_data_splits(f'logs/scotus/issueArea_civil_rights/{a}/{s}/', camel_case_pg)
                if df is None:
                    continue
                
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, ):  # more options can be specified also
                        print(df.to_string(index=False))
                print()