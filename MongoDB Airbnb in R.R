#installing and loading the mongolite library to download the Airbnb data
install.packages("mongolite") #need to run this line of code only once and then you can comment out
library(mongolite)

# This is the connection_string. You can get the exact url from your MongoDB cluster screen
#replace the <<user>> with your Mongo user name and <<password>> with the mongo password
#lastly, replace the <<server_name>> with your MongoDB server name
connection_string <- 'mongodb+srv://<<user>>:<<password>>@<<server_name>>'
airbnb_collection <- mongo(collection="listingsAndReviews", db="sample_airbnb", url=connection_string)

#Here's how you can download all the Airbnb data from Mongo
## keep in mind that this is huge and you need a ton of RAM memory

airbnb_all <- airbnb_collection$find()

#######################################################
#if you know or want to learn MQL (MongoQueryLanguage), that is a JSON syntax, feel free to use the following:::
######################################################
#1 subsetting your data based on a condition:
mydf <- airbnb_collection$find('{"bedrooms":2, "price":{"$gt":50}}')

#2 writing an analytical query on the data::
mydf_analytical <- airbnb_collection$aggregate('[{"$group":{"_id":"$room_type", "avg_price": {"$avg":"price"}}}]')
