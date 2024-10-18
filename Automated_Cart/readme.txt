1. The file create.sql is used to create all the tables in the database.
2. The file load.sql is used to load all the values for the tables. 
	This file uses the attached two data files, barcode_data.csv and data.csv.
	You have to specify the exact absolute path of these two files in the load.sql before running it.
	This is because copy command needs the exact path.
3. The third file is the pg_dump file that contains both the tables and the data.
	It is highly recommended to use this file for table creation and data loading.
	
######################### NOTE #########################

IT IS ABSOLUTELY IMPORTANT THAT THE NAME OF THE DATABASE IS EXACTY
"Automated_Cart" FOR THE PROJECT TO RUN PROPERLY.

This is because the settings.py file in Django has been configured to connect to this exact database name.