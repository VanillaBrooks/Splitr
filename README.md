# Splitr
Read printed reciepts and partition expenses to various people


## Inspiration
When splitting costs between friends it is common to have all items on a single reciept and partition the costs at a later date. However, It is very time consuming to type product names into an invoice or numbers into a calculator. Splitr alleviates this burden by automatically parsing the names of products and their associated costs. With this advent, a user only needs to assign items to people instead of perform trivial calculations. 

## Usage 

The code is packaged into a .apk to be installed on an andorid phone. The reciept is scanned using the android camera API. Each line of the reciept is parsed and sent to the OCR model to parse characters. The cost associated with each item on the reciept is then partitioned to different users and payment is requested using the Venmo API. If the OCR model is not confident in its result for an item, the user has the option to input the value manually.

The account information (venmo, phone #s) of users is stored between uses to further reduce the amount of time wasted. 

## Features

Splitr also stores information on items that were purchased to be used at a later date for data representations such as amount spent over time, money spent in different categories, or purchasing history with different friends. 
