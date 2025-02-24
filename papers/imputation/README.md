## Summary of papers (from my own perspective only, no AI)  
### What imputation methods will you recommend to use after reading these 4 papers?   
Generally, I will recommend to use **tree-based methods**, especially random forest, as a first method to try. Since, it's effective and less computation. To go into more detail, I'll suggest to know the downstream task first. **If the downstream is regression task, RNN-based methods is better; if the downstream is forecasting task, using tree-based methods is enough.**   
### What metrics do researcher commonly use to evaluate the imputation performace?  
MAE, MSE, and NRMSE  
### What's different between regression and forecasting?  
Regression aims at establishing a mathematical relationship between the input variables and the corresponding output variable. Forecasting's primary goal is to predict future values based on temporal patterns, supervised regression aims to understand and quantify the relationships between variables. (source: https://www.amorphousdata.com/blog/time-series-vs-regression)  
