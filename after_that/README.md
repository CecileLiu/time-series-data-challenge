# What Should You Do  

## Exploratory Data Analysis  
  
### Context  
During the test, I use "cta-ridership-l-station-entries-daily-totals.csv" file only. I feel the number of features is not enough to solve the second question, the drawback of the infrastructure.  
So, if I have had more time, the first thing I'd like to do is to explore other csv files.  
#### Steps  
1. look carefully to the Ridership files:
    ```
    cta-ridership-bus-routes-daily-totals-by-route.csv
    cta-ridership-daily-boarding-totals.csv
    cta-ridership-l-station-entries-daily-totals.csv
    ```
2. There are 3 files contain **daily** information
3. 2 of the above files contain each transportation station information
4. Combine (concat vertically) the above 2 file `cta-ridership-bus-routes-daily-totals-by-route.csv` and `cta-ridership-l-station-entries-daily-totals.csv`.  
5. Right now I have data of number of rides per station (both bus and train) per day.    
  
### Context  
During the test, I didn't raise foundamental questions for paving the road to the final goal, "the drawback of the infrastructure".  
So, if I have had more time, I'd like to see what questions I can raise from the combined data.   
#### Steps  
1. The columns of the combined data are: `date`, `id`, `daytype`, `rides` and `transportation`.   
When `transportation` is "bus", then the value of `id` means `route`; when the `transportation` is `train`, then the value of `id` means `station_id`. 
2. Don't forget we have another data, `cta-ridership-daily-boarding-totals.csv`  
3. Raise questions:
    * How does the number of rides vary by day type (e.g., weekday, weekend, holiday) across different transportation modes (bus vs. train)?
    * Are there any seasonal trends in the number of rides for both buses and trains?
    * Which bus and train stations have the highest and lowest ridership numbers over the years?
