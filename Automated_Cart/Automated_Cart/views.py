from django.shortcuts import render,redirect
from django.contrib import messages
import psycopg2 as pg
from django.core.mail import send_mail
import socket

def checkout(request):
    flag = 1
    
    if request.POST:
        ip = f'http://{get_ip()}:8000/'
        print(ip)
        flag = 0
        send_mail(
        'Order Confirmed',
        f'Dear {request.session["username"]},\nYour order is confirmed.\nThank you for shopping with Walmart.Regards.',
        'hrishikesh.vichore16@siesgst.ac.in',
        [request.session["username"]],
        fail_silently=True,
        )
        sql = """ Select id from auth_user where email = %s"""
        result, _ = executeQuery(sql,(request.session["username"],))
        result = result[0][0]
        
        sql = '''alter table cart_items enable trigger ALL;'''
        executeQuery(sql)
        sql = f'DELETE from cart_items where uid = {result};'
        
        executeQuery(sql)
        request.session.flush()
        return render(request,'checkout.html', {'flag':flag,'ip':ip})
    return render(request,'checkout.html', {'flag':flag})

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def executeQuery(sql, data = None):
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
        conn.commit()
        cur.close()
        conn.close()
        return e, False
                
def sign_up_page(request):
    if request.POST:
        if request.POST.get('login_btn') == 'Sign up':
                
            data = (request.POST.get('pwd'), request.POST.get('email'), 
                    request.POST.get('fname'), request.POST.get('lname'), request.POST.get('email'))
            
            sql = "insert into auth_user(password, username, first_name, last_name, email) values \
            (%s,%s,%s,%s,%s) RETURNING ID;"
            result, flag = executeQuery(sql, data)
            
            if flag:
                return redirect('/accounts/login/')
            else:
                return render(request,'sign_up.html', {'error_msg':'User already exists.\nTry using a different email.'})
            
    return render(request, 'sign_up.html')

def formView(request):
    request.session.clear_expired()
    
    if request.session.has_key('username'):
        
        return redirect('/cart/')
    else:
        
        return render(request,'registration/login.html')


def login_check(request):
    
    if request.POST:
        email = request.POST.get('email')
        pwd = request.POST.get('pwd')
        sql = 'select 1 from auth_user where email = %s'
        data = (email,)
        result, flag = executeQuery(sql, data)
        if flag and len(result)>0:
            sql = 'select password from auth_user where email = %s'
            data = (email,)
            result, flag = executeQuery(sql, data)
            if flag:
                if result[0][0] == pwd:
                    sql = 'Update auth_user set last_login = CURRENT_TIMESTAMP WHERE email = %s'
                    data = (email,)
                    executeQuery(sql = sql, data = data)
                    request.session['username'] = email
                    request.session.set_expiry(600)
                    return redirect('/cart/')
            
                else:
                    return render(request,'registration/login.html', {'error_msg':'Password is incorrect'})
            
            else:
                return render(request,'registration/login.html', {'error_msg':result})
        else:
            return render(request,'registration/login.html', {'error_msg':"Email doesn't exist"})

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

        
def cart_items(request):
    if request.session.has_key('username'):
        flag = 0
        if request.POST:
        
            if request.POST.get('b1') == 'barcode':
                    ip = get_client_ip(request)
                    request.session['ip'] = ip
                    barcodeScanner(request.session['username'],request.session['ip'])
                                        
            elif request.POST.get('b1') == 'search': 
                product_name =  request.POST.get('search_box').lower().capitalize()
                
                sql = ''' Select product_name, package_size, price, brand, pid from items where product_name @@ %s;'''
                result, _ = executeQuery(sql, (product_name,))  
                result = enumerate(result)
                return render(request,'main.html',{'product': result, 'search_product':1})
            
            elif request.POST.get('b1') == 'add_selected':
                products = tuple(request.POST.getlist('product_list'))
                for pid in products:
                    sql = "select pid from cart_items c1 where c1.pid = %s;"
                    result, _ = executeQuery(sql, (pid,))
                    if not len(result):
                        sql = """ Select id from auth_user where email = %s"""
                        result, _ = executeQuery(sql,(request.session['username'],))
                        result = list(result[0])
                        sql = """Select pid, product_name, price
                                from items where pid = %s"""
                        result.extend(list(executeQuery(sql, (pid,))[0][0]))
                        
                        sql = '''insert into cart_items(uid,pid, product_name, price) values
                        (%s, %s, %s, %s) RETURNING cart_items.pid;''' 
                        executeQuery(sql, tuple(result))
                    else:
                        sql = '''update cart_items c1 set quantity = quantity + 1 where c1.pid = %s RETURNING c1.pid;''' 
                        executeQuery(sql, (pid,))
                
            elif request.POST.get('b1') == 'checkout':
                return redirect('/checkout/')
            
            elif request.POST.get('b1') == 'remove_item':
                items_to_remove = tuple(request.POST.getlist('cart_items'))
                sql = '''alter table cart_items disable trigger ALL;'''
                executeQuery(sql)
                for pid in items_to_remove:
                    
                    sql = '''Delete from cart_items c1 where c1.pid = %s returning 1;'''
                    executeQuery(sql, (pid,))
            elif request.POST.get('b1') == 'logout':
                request.session.clear()
                request.session.flush()
                return redirect('/check_login/')
            
            sql = 'select product_name, price, quantity, pid from cart_items where uid=(select id from auth_user where email = %s);'
            result, _ =  executeQuery(sql, (request.session['username'],))
            
            if len(result):
                flag = 1
                sql = 'select count(1), sum(quantity*price), sum(quantity) from cart_items where uid=(select id from auth_user where email = %s);'
                items, price, quantity = executeQuery(sql, (request.session['username'],))[0][0]
                return render(request,'main.html',{'found': result,"l": items, "p": price, "w": quantity,
                            'flag': flag})
            
        return render(request,'main.html')
    else:
        return redirect('/check_login/')


def barcodeScanner(username,ip):
    from pyzbar import pyzbar
    import cv2
        
    link = f"http://{ip}:8080/video"
    cam = cv2.VideoCapture(link)
    while cam.isOpened():
        _, frame = cam.read()
        frame = cv2.resize(frame, (400,400))
        cv2.imshow("Barcode Scanner", frame)
        cv2.setWindowProperty("Barcode Scanner",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty("Barcode Scanner",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_NORMAL)
        _ = cv2.waitKey(1) & 0xFF
        barcodes = pyzbar.decode(frame)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = barcode.data.decode("utf-8")
            sql = "select pid from cart_items c1 where c1.pid = (select pid from barcodes_table where barcode = '%s');" %barcodeData
            result, _ = executeQuery(sql)
            if not len(result):
                sql = """ Select id from auth_user where email = %s"""
                result, _ = executeQuery(sql,(username,))
                result = list(result[0])
                sql = """Select pid, product_name, price
                        from items i1 where i1.pid=(select pid from barcodes_table where barcode = '%s')"""%barcodeData
                result.extend(list(executeQuery(sql)[0][0]))
                
                sql = '''insert into cart_items(uid,pid, product_name, price) values
                (%s, %s, %s, %s) RETURNING cart_items.pid;''' 
                
                executeQuery(sql, tuple(result))
                
            else:
                sql = """ Select id from auth_user where email = %s"""
                result, _ = executeQuery(sql,(username,))
                result = result[0][0]
                
                sql = f'''update cart_items c1 set quantity = quantity + 1 where c1.pid = (select pid from cart_items c1 where c1.pid = (select pid from items i1 where i1.pid=(select pid from barcodes_table where barcode = '{barcodeData}'))) AND uid = {result} RETURNING c1.pid;'''
                
                executeQuery(sql)
            cam.release()
            cv2.destroyAllWindows()
            return 1