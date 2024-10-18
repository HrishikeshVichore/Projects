CREATE TABLE if not exists ITEMS(
	PRODUCT_NAME VARCHAR(500) NOT NULL,
	PRICE DECIMAl,
	BRAND VARCHAR(500),
	PACKAGE_SIZE VARCHAR(100),
	PID integer primary key
);

CREATE TABLE IF NOT EXISTS auth_user
(   id serial,
    password varchar (128) NOT NULL,
    last_login timestamp with time zone,
    is_superuser boolean NOT NULL DEFAULT false,
    username varchar(150) NOT NULL,
    first_name varchar(150)NOT NULL,
    last_name varchar(150)  NOT NULL,
    email varchar(254) NOT NULL,
    is_staff boolean NOT NULL DEFAULT false,
    is_active boolean NOT NULL DEFAULT true,
    CONSTRAINT auth_user_pkey PRIMARY KEY (id),
    CONSTRAINT auth_user_username_key UNIQUE (username),
    CONSTRAINT unique_email UNIQUE (email)
);

Create table if not exists cart_items(
	UID integer not null,
	PID integer not null,
	PRODUCT_NAME VARCHAR(500) NOT NULL,
	PRICE DECIMAl,
	QUANTITY integer default 1,
	FOREIGN KEY (PID) REFERENCES items(PID),
	FOREIGN KEY (UID) REFERENCES auth_user(ID),
	Constraint pk_cart_items Primary key(UID,PID)
);

CREATE TABLE IF NOT EXISTS public.django_session
(
    session_key varchar(40) NOT NULL,
    session_data text NOT NULL,
    expire_date timestamp with time zone NOT NULL,
    CONSTRAINT django_session_pkey PRIMARY KEY (session_key)
);

create table if not exists barcodes_table(
	pid integer primary key,
	barcode VARCHAR(100) unique
);

Create table if not exists order_history(
	UID integer not null,
	OID varchar(100) primary key,
	FOREIGN KEY (UID) REFERENCES auth_user(ID)
);

Create table if not exists purchase_history(
	OID varchar(100) not null,
	pid integer not null,
	price DECIMAL,
	quantity integer,
	FOREIGN KEY (PID) REFERENCES items(PID)
);

create or replace function insert_into_purchase_history()
Returns trigger
AS $create_purchase_history$
declare oid varchar(100) = concat(to_char(current_timestamp, 'YYYY/MM/DD-HH12:MI:SS'),'_',old.uid);
begin 
insert into purchase_history values (oid, old.pid, old.price, old.quantity);
return null;
END;
$create_purchase_history$ Language plpgsql;



create trigger create_purchase_history
after delete on cart_items
for each row 
execute procedure insert_into_purchase_history();


create or replace function insert_into_order_history()
Returns trigger
AS $create_order_history$
declare uid int = (select distinct uid from old_table);
declare oid varchar(100) = concat(to_char(current_timestamp, 'YYYY/MM/DD-HH12:MI:SS'),'_',uid);
begin 
insert into order_history values(uid,oid);
return null;
END;
$create_order_history$ Language plpgsql;

--drop trigger if exists create_order_history;

create trigger create_order_history
after delete on cart_items
referencing old table as old_table
for each statement
execute procedure insert_into_order_history();


