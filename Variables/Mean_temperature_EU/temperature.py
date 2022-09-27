import pandas as pd

#read csv file of the countries
country = pd.read_csv('malta.csv', sep = ',') #select the country's name file in csv

country.info() 
print('Starting data missing values')
print(country.isnull().sum())

#Some csv file were jointed, hence there might be multiple 
country = country.drop(country[country['DATE'].str.len() < 10].index)

#transform the Dtype
country['DATE'] = pd.to_datetime(country['DATE'])
country['PRCP'] = pd.to_numeric(country['PRCP'])
country['SNWD'] = pd.to_numeric(country['SNWD'])
country['TAVG'] = pd.to_numeric(country['TAVG'])
country['TMAX'] = pd.to_numeric(country['TMAX'])
country['TMIN'] = pd.to_numeric(country['TMIN'])

#missing values:
#   - For null values in average temperature, we use the mean between max and min temperature when both available
#   - If a station has more than 365 days (out of 365x7) with null values for the average temperature, we drop the entire station

country['TAVG'] = country['TAVG'].fillna(country[['TMAX','TMIN']].mean(axis=1, skipna=False))

print('Fill missing values of TAVG with the mean between TMAX and TMIN when possible')
print(country.isnull().sum())

country = country.groupby('STATION').filter(lambda x: x.TAVG.notnull().sum()>2192)

print('Missing values of data without stations with more than 365 missing values')
print(country.isnull().sum())

country.info() 
#check for having daily observations in each s
country = (country.set_index('DATE')
        .groupby(['STATION', 'NAME'], sort=False)['PRCP','SNWD','TAVG','TMAX','TMIN']
        .apply(lambda x: x.reindex(pd.date_range('2015-01-01', '2021-12-31',
                                                 name='DATE'), method='ffill'))
        .reset_index())

#sort by date to obtain a national daily average

country = country.sort_values(by="DATE")

country_daily_averages = country.groupby(pd.Grouper(key='DATE',freq='1D')).mean()

#country_daily_averages.to_csv('latvia_daily_averages.csv', sep=',')

# croatia, denmark, finland, netherlands, sweden, uk have a lot of missing dates
# latvia has all mean temperatures, but few days. Hence we first add the days and then filter