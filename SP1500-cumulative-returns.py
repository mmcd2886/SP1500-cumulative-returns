import pandas as pd
import warnings # hide deprecation warnings for now

warnings.filterwarnings('ignore')


# run groupby function. Sum the transactionShares and Weight columns
def groupby_weight(form4_csv):
    # create weight column
    form4_csv['Weight'] = form4_csv['transactionShares'] * form4_csv['transactionPricePerShare']
    # groupby different columns. The columns are sorted with 'sharesOwnedFollowingTransaction' from least to greatest
    # (this is done in the main section).when you do the groupby, you want to get the last value of
    # 'sharesOwnedFollowingTransaction' for that group which is the greatest .this will be the amount of shares owned
    # after all the purchases. This allows you to add all the shares purchased for that insider on a particular day and
    # then compare that to how many shares the insider owns at the end of day.
    form4_csv_shares_following = form4_csv.loc[
        form4_csv.groupby(['transactionDate', 'rptOwnerName', 'rptOwnerCik', 'documentType', 'issuerCik', 'issuerName',
                           'issuerTradingSymbol', 'transactionCode', 'securityTitle'], as_index=False)[
            'sharesOwnedFollowingTransaction'].idxmax()]
    # sum the 'transactionShares' and 'Weight' columns during the groupby operation
    form4_csv_weight = \
        form4_csv.groupby(['transactionDate', 'rptOwnerName', 'rptOwnerCik', 'documentType', 'issuerCik', 'issuerName',
                           'issuerTradingSymbol', 'transactionCode', 'securityTitle'], as_index=False)[
            'transactionShares', 'Weight'].sum()
    # drop all columns from form4_csv_shares_following except for 'sharesOwnedFollowingTransaction' then reset index
    form4_csv_shares_following = form4_csv_shares_following[['sharesOwnedFollowingTransaction']].reset_index(drop=True)
    form4_csv_weight = form4_csv_weight.reset_index(drop=True)
    # now join the two dataframes on the indexes. This will append the 'sharesOwnedFollowingTransaction' column
    form4_csv = pd.merge(form4_csv_weight, form4_csv_shares_following, left_index=True, right_index=True).reset_index()
    # subtract the the 'totalSharesPurchased' column from the 'sharesOwnedFollowingTransaction' to get the amount of
    # shares owned by insider prior to making any purchases.
    form4_csv['sharesOwnedPriorToPurchases'] = form4_csv['sharesOwnedFollowingTransaction'] - form4_csv[
        'transactionShares']
    # calculate the percentage increase of shares for insider
    form4_csv['percentageIncreaseShares'] = (form4_csv['sharesOwnedFollowingTransaction'] - form4_csv[
        'sharesOwnedPriorToPurchases']) / form4_csv['sharesOwnedPriorToPurchases']
    # Divide the 'Weight' column by the 'transactionShares' to calculate the weighted average
    form4_csv['Weighted Average'] = form4_csv['Weight'] / form4_csv['transactionShares']
    # create total purchase dollar amount column
    form4_csv['Total Purchase'] = form4_csv['Weighted Average'] * form4_csv['transactionShares']
    # Delete the 'Weight' column as it is no longer needed
    form4_csv = form4_csv.drop(['Weight', 'sharesOwnedPriorToPurchases'], axis=1)
    # to csv
    return form4_csv


def date_offset_func(date_of_form4_purchase, ticker):
    # return a dataframe that contains stock info for company with matching CIK and Ticker
    date_offset_stock_df = yahoo_historical_csv[(yahoo_historical_csv['Symbol'] == ticker)]
    # convert datettime object to string
    date_of_form4_purchase = date_of_form4_purchase.strftime("%Y-%m-%d")
    # locate row with matching date
    date_df = date_offset_stock_df[(date_offset_stock_df['Date'] == date_of_form4_purchase)]
    # if no stock data is found exit loop
    if date_df.empty:
        return None
    else:
        pass
    # get the index number for the date of the purchase
    index_number = date_df.index.item()
    index_180 = index_number + 783
    # create new df that has stock data from 0-180 days from purchase
    new_df = date_offset_stock_df.loc[index_number:index_180]
    new_df = new_df.reset_index(drop=True)
    # if 1095 days of stock data is not found for ticker & cik, go to next transaction
    if len(new_df.index) < 784:
        return None
    return new_df


def returns_func(df):
    df['Stock Daily Return'] = df[['Stock Adj Close']].pct_change()
    daily_returns = df['Stock Daily Return']
    # calculate mean, standard deviation, cumulative return
    df['Stock Mean Daily Returns'] = daily_returns.expanding().mean()
    df['Stock SD Daily Returns'] = daily_returns.expanding().std()
    df['Stock Cumulative Returns'] = (df['Stock Daily Return'] + 1).cumprod() - 1
    return df


def sp1500_returns_func(sp1500_csv, date_of_form4_purchase):
    number_of_days = 783
    # convert datettime object to string
    date_of_form4_purchase = date_of_form4_purchase.strftime("%Y-%m-%d")
    # find row with matching date
    sp1500_csv_date = sp1500_csv[(sp1500_csv['Date'] == date_of_form4_purchase)]
    # get the index number for the date of the purchase
    try:
        index_number = sp1500_csv_date.index.item()
    except:
        display('!!! FAIL')
        return
    index_180 = index_number + 783
    # create new df that has stock data from 0-180 days from purchase
    sp1500_csv = sp1500_csv.loc[index_number:index_180]
    sp1500_csv = sp1500_csv.reset_index(drop=True)

    # calculate daily returns
    sp1500_csv['SP1500 Daily Return'] = sp1500_csv[['SP1500 Adj Close']].pct_change()
    daily_returns = sp1500_csv['SP1500 Daily Return']
    # calculate mean, standard deviation, cumulative return
    sp1500_csv['SP1500 Mean Daily Returns'] = daily_returns.expanding().mean()
    sp1500_csv['SP1500 SD Daily Returns'] = daily_returns.expanding().std()
    sp1500_csv['SP1500 Cumulative Returns'] = (sp1500_csv['SP1500 Daily Return'] + 1).cumprod() - 1
    return sp1500_csv


def merge_form4_sp1500_func(sp_1500_returns_df, days_returns_df):
    merged_df = pd.merge(days_returns_df, sp_1500_returns_df, how='inner', left_index=True, right_index=True)
    # calculate abnormal returns
    abnormal_return = merged_df['Stock Daily Return'] - merged_df['SP1500 Daily Return']
    merged_df['Stock Daily Abnormal Return'] = abnormal_return
    # calculate abnormal return for cumulative returns
    cumulative_abnormal_return = merged_df['Stock Cumulative Returns'] - merged_df['SP1500 Cumulative Returns']
    merged_df['Stock Abnormal Cumulative Returns'] = cumulative_abnormal_return
    #     merged_df['Stock Abnormal Cumulative Returns'] = (merged_df['Stock Daily Abnormal Return'] + 1).cumprod() -1
    return merged_df


def organize_columns(merge_form4_sp1500_df, rptOwnerName, documentType, rptOwnerCik, totalPurchase,
                     percentageIncreaseShares):
    # add columns to dataframe
    merge_form4_sp1500_df['documentType'] = documentType
    merge_form4_sp1500_df['rptOwnerCik'] = rptOwnerCik
    merge_form4_sp1500_df['rptOwnerName'] = rptOwnerName
    merge_form4_sp1500_df['Total Purchase'] = totalPurchase
    merge_form4_sp1500_df['percentageIncreaseShares'] = percentageIncreaseShares
    # organize columns of dataframe
    full_csv = merge_form4_sp1500_df[
        ['Date_x', 'documentType', 'rptOwnerName', 'rptOwnerCik', 'Symbol', 'CIK', 'Total Purchase',
         'percentageIncreaseShares',
         'Stock Daily Return', 'Stock Cumulative Returns', 'Stock Daily Abnormal Return',
         'Stock Abnormal Cumulative Returns'
         ]]
    return full_csv


def single_row(full_df):
    # create a new dataframe from full_df that has column for date, rptownername,symbol,cik,and cumul. returns for
    # 1,3,7,21,90,80
    empty_frame = pd.DataFrame()
    # create additional columns and append the value for the column using iloc
    empty_frame['Date'] = [full_df.iloc[0, 0]]
    empty_frame['rptOwnerName'] = [full_df.iloc[0, 2]]
    empty_frame['rptOwnerCik'] = [full_df.iloc[0, 3]]
    empty_frame['Symbol'] = [full_df.iloc[0, 4]]
    empty_frame['CIK'] = [full_df.iloc[0, 5]]
    empty_frame['Total Purchase'] = [full_df.iloc[0, 6]]
    empty_frame['percentageIncreaseShares'] = [full_df.iloc[0, 7]]

    # 1 day, 3 day, 1 week, 3 week, 3 months, 6 months, 1 year , 3 year
    days_offset = [1, 3, 6, 16, 64, 130, 261, 783]
    # append cumulative returns
    for days in days_offset:
        location = full_df.iloc[days, 9]
        empty_frame['Cumulative Returns', days] = [location]
    # append abnormal returns
    for days in days_offset:
        location = full_df.iloc[days, 11]
        empty_frame['Abnormal Returns', days] = [location]
    return empty_frame


# import yahoo_historical, form4 and ticker csv's
yahoo_historical_csv = pd.read_csv('/Users/a1/Desktop/from_scratch/all_tickers/sp1500_yahoo_historical.csv')
form4_csv = pd.read_csv('/Users/a1/Desktop/from_scratch/nonDerivative/all_nonDerivative_purchases_2009_2019.csv')
sp1500_csv = pd.read_csv('/Users/a1/Desktop/from_scratch/all_tickers/sp1500tr_index.csv')

# create a new sp1500 dataframe that only contains necessary columns
sp1500_csv = sp1500_csv[['Date', 'Close']]
# rename 'Adj Close' column to 'sp1500 Adj Close'
sp1500_csv = sp1500_csv.rename(columns={'Close': 'SP1500 Adj Close'})

# create a new yahoo_historical dataframe with only the necessary columns
yahoo_historical_csv = yahoo_historical_csv[['Date', 'Symbol', 'Adj Close']]
# rename 'Adj Close' column to 'Stock Adj Close'
yahoo_historical_csv = yahoo_historical_csv.rename(columns={'Adj Close': 'Stock Adj Close'})
# convert 'Date' to datetime object
yahoo_historical_csv['Date'] = pd.to_datetime(yahoo_historical_csv['Date'])
# convert all ticker symbols to uppercase
yahoo_historical_csv['Symbol'] = yahoo_historical_csv['Symbol'].str.upper()
# convert 'transactionDate' to datetime object
form4_csv['transactionDate'] = pd.to_datetime(form4_csv['transactionDate'])

# filter all transactions for only directors or officers
#
####IMPORTANT 
#
# These Numbers may need to be changed to ints depending on the csv
form4_csv = form4_csv[(form4_csv['isDirector'] == 1) | (form4_csv['isOfficer'] == 1)]

index = form4_csv.index
number_of_rows = len(index)
print(number_of_rows)

# sort the form4 csv by transactionDate and then by rptOwnerName, then by sharesOwnedFollowingTransaction
form4_csv = form4_csv.sort_values(by=['transactionDate', 'rptOwnerName', 'sharesOwnedFollowingTransaction'])
# run groupby function on entire form4 csv
form4_csv = groupby_weight(form4_csv)
# convert all ticker symbols to uppercase
form4_csv['issuerTradingSymbol'] = form4_csv['issuerTradingSymbol'].str.upper()

# iterate over rows of csv for each transaction. extract the date, tradingSymbol and CIK
for index, row in form4_csv.iterrows():
    # return a dataframe that contains 180 days of stock data from tx date that also matches cik & ticker on form 4
    date_offset_stock_df = date_offset_func(row['transactionDate'], row['issuerTradingSymbol'])
    # if there is no stock data for the transaction, go to the next transaction
    if date_offset_stock_df is None:
        continue
    # Append the CIK to yahoo_historical data. This differs from 'stock_analytics.pynb' version. The yahoo_historical
    # data does not have cik numbers already in columns in this method
    date_offset_stock_df.insert(1, 'CIK', row['issuerCik'])
    # calculate returns data for stock
    tx_returns_df = returns_func(date_offset_stock_df)
    # return a dataframe with returns data for sp1500 starting with transactionDate through 180 days
    sp1500_returns_df = sp1500_returns_func(sp1500_csv, row['transactionDate'])
    # merge the sp1500 returns df with the stock returns df
    merge_form4_sp1500_df = merge_form4_sp1500_func(sp1500_returns_df, tx_returns_df)
    # add columns to merge_form4_sp1500_df
    full_df = organize_columns(merge_form4_sp1500_df, row['rptOwnerName'], row['documentType'], row['rptOwnerCik'],
                               row['Total Purchase'], row['percentageIncreaseShares'])
    # organize to a single row
    try:
        final_full_df = single_row(full_df)
    except Exception as e:
        (full_df, e)
        continue
    # output to csv
    final_full_df.to_csv('/Users/a1/Desktop/from_scratch/all_tickers/final.csv', mode='a', header=False, index=False)
