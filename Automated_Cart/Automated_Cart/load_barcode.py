from random import randint
from barcode import Code128
from barcode.writer import ImageWriter
import psycopg2 as pg
import os
import shutil

def executeQuery(sql, data):
    try:
        conn = pg.connect(
            host = 'localhost',
            database = 'Automated_Cart',
            user = 'postgres',
            password = '1234',
            port = 5432,
            )
        cur = conn.cursor()
        
        cur.execute(sql, data)
        
        result = cur.fetchall()
        
        conn.commit()
        cur.close()
        conn.close()
        return result, True
    except Exception as e:
        cur.close()
        conn.close()
        return e, False
    
def create_barcodes(delete_existing = False):
    if delete_existing:
        shutil.rmtree(folder_name)
    sql = 'select count(*) from items;'
    data = ()
    result, _ = executeQuery(sql, data)
    
    barcodes = []
    n = 12
    range_start = 10**(n-1)
    range_end = (10**n)-1
    while len(barcodes)<result[0][0]:
        x = randint(range_start, range_end)
        if x not in barcodes:
            barcodes.append(x)
            my_code = Code128(str(x), writer=ImageWriter())
            my_code.save(f'barcode_folder/{str(x)}')
    print(len(barcodes))


def insert_barcodes():
    print('Inserting barcodes into database')
    barcodes = os.listdir('barcode_folder')
    barcodes = [i.split('.')[0] for i in barcodes]
    print(len(barcodes), barcodes[0])
    
    conn = pg.connect(
                host = 'localhost',
                database = 'Automated_Cart',
                user = 'postgres',
                password = '1234',
                port = 5432,
                )
    cur = conn.cursor() 
    
    
    sql = 'update barcodes_table set barcode = %s where barcodes_table.pid=%s;'
    barcodes = list(enumerate(barcodes))
    barcodes = [(b,a) for a,b in barcodes]
    cur.executemany(sql, barcodes)  
    print('Inserted')
    conn.commit()
    cur.close()
    conn.close()
    
if __name__ == '__main__':
    
    CREATE_BARCODES = False
    delete_existing = CREATE_BARCODES
    folder_name = 'barcode_folder'
    
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
        CREATE_BARCODES = True
    
    if not len(os.listdir(folder_name)):
        CREATE_BARCODES = True
    
    if CREATE_BARCODES:
        create_barcodes(delete_existing)

    insert_barcodes()

