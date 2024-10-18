SET CLIENT_ENCODING TO 'utf8';

copy ITEMS(PRODUCT_NAME,PRICE,BRAND,PACKAGE_SIZE,PID) 
from 'D:\Hrishikesh_Vichore\CSE 560 DMQL\Project\Final_Project_Work\data.csv'
with(
	FORMAT csv, header
);

copy barcodes_table(PID,BARCODE) 
from 'D:\Hrishikesh_Vichore\CSE 560 DMQL\Project\Final_Project_Work\barcode_data.csv'
with(
	FORMAT csv, header
);
